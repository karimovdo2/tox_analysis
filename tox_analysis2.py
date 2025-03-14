import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

###############################################################################
# ФУНКЦИИ ДЛЯ АНАЛИЗА (адаптированные из вашего Tkinter-кода)
###############################################################################

def transliterate_column_name(s: str) -> str:
    mapping = {
        'А':'A','а':'a','Б':'B','б':'b','В':'V','в':'v','Г':'G','г':'g','Д':'D','д':'d','Е':'E','е':'e',
        'Ж':'Zh','ж':'zh','З':'Z','з':'z','И':'I','и':'i','Й':'J','й':'j','К':'K','к':'k','Л':'L','л':'l',
        'М':'M','м':'m','Н':'N','н':'n','О':'O','о':'o','П':'P','п':'p','Р':'R','р':'r','С':'S','с':'s',
        'Т':'T','т':'t','У':'U','у':'u','Ф':'F','ф':'f','Х':'H','х':'h','Ц':'Ts','ц':'ts','Ч':'Ch','ч':'ch',
        'Ш':'Sh','ш':'sh','Щ':'Shch','щ':'shch','Ы':'Y','ы':'y','Э':'E','э':'e','Ю':'Yu','ю':'yu','Я':'Ya','я':'ya'
    }
    for ru, la in mapping.items():
        s = s.replace(ru, la)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    return s

def replace_outliers_in_PC1(df, group_col, col, threshold=2.0, groups_to_clean=None):
    df_new = df.copy()
    for group, subdf in df_new.groupby(group_col):
        if groups_to_clean is not None and group not in groups_to_clean:
            continue
        median_val = subdf[col].median()
        mu = subdf[col].mean()
        sigma = subdf[col].std()
        if sigma == 0:
            continue
        outliers = subdf.index[np.abs((subdf[col] - mu) / sigma) > threshold]
        df_new.loc[outliers, col] = median_val
    return df_new

def format_sci_custom(num, precision=1):
    """Форматируем число в виде 1,0×10⁻³ и т.д."""
    s = f"{num:.{precision}e}"
    s = s.replace('.', ',')
    base, exp = s.split('e')
    exp = exp.lstrip('+')
    superscript = {
        '-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵',
        '6':'⁶','7':'⁷','8':'⁸','9':'⁹'
    }
    exp_str = ''.join(superscript.get(ch, ch) for ch in exp)
    if base == '1,0':
        return f"10{exp_str}"
    else:
        return f"{base}×10{exp_str}"

###############################################################################
# МОДЕЛИ
###############################################################################
def model_linear(d, a, b):
    return a + b*d

def model_quadratic(d, a, b, c):
    return a + b*d + c*(d**2)

def model_log(d, a, b):
    return a + b*np.log(d + 1.0)

def model_exp(d, a, b, c):
    return a + b*np.exp(c*d)

def model_power(d, a, b, c):
    return a + b*(d**c)

def compute_aic(n, rss, k):
    val = rss/n + 1e-15
    ll = n * np.log(val)
    return ll + 2*k

def fit_and_evaluate(model_func, dose, pc1, p0):
    try:
        popt, _ = curve_fit(model_func, dose, pc1, p0=p0, maxfev=100000)
        pred = model_func(dose, *popt)
        mse = mean_squared_error(pc1, pred)
        rss = np.sum((pc1 - pred)**2)
        n = len(pc1)
        k = len(popt)
        aic = compute_aic(n, rss, k)
        return {"params": popt, "MSE": mse, "AIC": aic, "success": True}
    except:
        return {"params": None, "MSE": math.inf, "AIC": math.inf, "success": False}

def find_dose_for_target(target_val, model_func, params,
                         d_min=0, d_max=1000, step=0.1):
    best_d = None
    min_diff = float('inf')
    for x in np.arange(d_min, d_max+step, step):
        val = model_func(x, *params)
        diff = abs(val - target_val)
        if diff < min_diff:
            min_diff = diff
            best_d = x
    return best_d

def risk_logistic_adj(d, A, B):
    """Скорректированная логистическая модель: r(0)=0, r(∞)=1"""
    f0 = 1.0 / (1.0 + np.exp(B*A))
    f_d = 1.0 / (1.0 + np.exp(-B*(d - A)))
    return (f_d - f0)/(1 - f0)

def find_dose_for_risk_adj(r, A, B):
    if r < 0 or r >= 1:
        return np.nan
    f0 = 1.0 / (1.0 + np.exp(B*A))
    f_target = r*(1 - f0) + f0
    if f_target <= 0 or f_target >= 1:
        return np.nan
    d_est = A - (1.0/B)*np.log((1.0/f_target)-1)
    return d_est

###############################################################################
# ОСНОВНАЯ ФУНКЦИЯ
###############################################################################
def run_analysis(file, remove_outliers, outlier_threshold, groups_text, risk_input):
    """Принимает:
       - file (UploadedFile объект или путь),
       - remove_outliers (bool),
       - outlier_threshold (float),
       - groups_text (строка вида '50,100'),
       - risk_input (float) — уровень риска.
       Возвращает словарь с графиками и текстовыми результатами.
    """
    # Читаем Excel из UploadedFile напрямую:
    df = pd.read_excel(file)

    # Транслитерация
    old_cols = df.columns.tolist()
    new_cols = [transliterate_column_name(col) for col in old_cols]
    df.columns = new_cols

    # Ищем столбец с дозой
    dose_col = None
    for c in df.columns:
        if c.lower() in ["doza","dose","доза","доза_"] or c.upper() == "DOZA":
            dose_col = c
            break
    if dose_col is None:
        return None, "Столбец с дозой не найден."

    df.rename(columns={dose_col:"Dose"}, inplace=True)
    all_cols = df.columns.tolist()
    predictors = [col for col in all_cols if col != "Dose"]

    # Заполняем пропуски
    for col in predictors:
        df[col].fillna(df[col].median(), inplace=True)

    # Удаляем столбцы, где нет значений
    grouped = df.groupby("Dose")
    cols_to_drop = []
    for col in predictors:
        if any(grouped[col].count() == 0):
            cols_to_drop.append(col)
    cols_to_drop = list(set(cols_to_drop))
    df.drop(columns=cols_to_drop, inplace=True)
    final_predictors = [col for col in predictors if col not in cols_to_drop]

    # PCA -> PC1
    X = df[final_predictors].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    df["PC1"] = X_pca[:,0]

    # Удаляем выбросы
    if remove_outliers:
        if groups_text.strip() == "":
            groups_to_clean = None
        else:
            try:
                groups_to_clean = [float(x.strip()) for x in groups_text.split(",")]
            except:
                return None, "Неверный формат групп доз (пример: 50,100)."
        df = replace_outliers_in_PC1(df, group_col="Dose", col="PC1",
                                     threshold=outlier_threshold,
                                     groups_to_clean=groups_to_clean)

    # Ищем лучшую модель
    dose_vals = df["Dose"].values
    pc1_vals = df["PC1"].values
    model_candidates = [
        ("Linear", model_linear, [0,0]),
        ("Quadratic", model_quadratic, [0,0,0]),
        ("Log", model_log, [0,0]),
        ("Exponential", model_exp, [0,1,0]),
        ("Power", model_power, [0,1,1])
    ]
    results = []
    for (mname, mfunc, p0) in model_candidates:
        r_ = fit_and_evaluate(mfunc, dose_vals, pc1_vals, p0)
        results.append((mname, r_))
    results_sorted = sorted(results, key=lambda x: x[1]["AIC"])
    best_name, best_res = results_sorted[0]
    best_func = [x for x in model_candidates if x[0]==best_name][0][1]
    best_params = best_res["params"]

    # График PC1
    fig_pc1, ax1 = plt.subplots()
    ax1.scatter(dose_vals, pc1_vals, alpha=0.6, label="Data PC1")
    dose_grid = np.linspace(dose_vals.min(), dose_vals.max(),200)
    pc1_pred = best_func(dose_grid, *best_params)
    ax1.plot(dose_grid, pc1_pred, 'r--', label=f"{best_name} fit")
    ax1.set_xlabel("Dose")
    ax1.set_ylabel("PC1")
    ax1.set_title("PC1 ~ Dose")
    ax1.legend()

    # BMD (5% прирост от PC1(0))
    pc1_0 = best_func(0, *best_params)
    target_pc1 = pc1_0 * 1.05
    bmd_5pct = find_dose_for_target(target_pc1, best_func, best_params,
                                    d_min=0, d_max=1000, step=0.1)

    # Empirical Risk
    df_control = df[df["Dose"]==0]
    if len(df_control)==0:
        # на случай, если нет группы Dose=0
        mu_c = df["PC1"].mean()
        sd_c = df["PC1"].std()
    else:
        mu_c = df_control["PC1"].mean()
        sd_c = df_control["PC1"].std()
    low_bound = mu_c - 2*sd_c
    up_bound = mu_c + 2*sd_c
    df["PC1_out"] = ~df["PC1"].between(low_bound, up_bound)
    risk_df = df.groupby("Dose")["PC1_out"].mean()

    # Если есть 0 в индексах, считаем, что риск = 0
    if 0 in risk_df.index:
        risk_df.loc[0] = 0.0

    fig_emp, ax2 = plt.subplots()
    ax2.plot(risk_df.index, risk_df.values*100, 'ro-')
    ax2.set_xlabel("Dose")
    ax2.set_ylabel("Risk (%)")
    ax2.set_title("Empirical Risk")

    # Adjusted Logistic Risk
    doses_risk = np.array(sorted(risk_df.index))
    risk_vals_emp = np.array([risk_df.loc[d_] for d_ in doses_risk])
    try:
        popt_log, _ = curve_fit(risk_logistic_adj, doses_risk, risk_vals_emp,
                                p0=[50,0.1], maxfev=100000)
        fig_log, ax3 = plt.subplots()
        d_grid_log = np.linspace(doses_risk.min(), doses_risk.max()*1.2, 150)
        risk_log_pred = risk_logistic_adj(d_grid_log, *popt_log)
        ax3.scatter(doses_risk, risk_vals_emp, c='r', label="Empirical Risk")
        ax3.plot(d_grid_log, risk_log_pred, 'g-', label="Adjusted Logistic fit")
        ax3.set_xlabel("Dose")
        ax3.set_ylabel("Risk (0..1)")
        ax3.set_title("Adjusted Logistic Risk Model")
        ax3.legend()
    except RuntimeError:
        popt_log = None
        fig_log, ax_ = plt.subplots()
        ax_.text(0.1,0.5,"Logistic model fitting failed.", fontsize=12)
        ax_.set_title("Adjusted Logistic Risk Model")

    # Тексты
    text_log = (
        "Скорректированная логистическая функция:\n"
        "r(d) = (f(d) - f(0)) / [1 - f(0)], где f(d) = 1/[1 + exp(-B(d - A))].\n\n"
    )
    user_risk = risk_input
    if popt_log is not None:
        user_risk_dose = find_dose_for_risk_adj(user_risk, *popt_log)
        text_log += f"Введённый риск: {user_risk:.5f} => Dose={user_risk_dose:.5f}\n\n"
        # Стандартные риски
        standard_risks = [1e-3, 1e-4, 1e-5]
        for rr in standard_risks:
            dd_ = find_dose_for_risk_adj(rr, *popt_log)
            r_str = format_sci_custom(rr, precision=1)
            text_log += f"Risk={r_str} => Dose={dd_:.5f} mg/kg/day\n"
    else:
        text_log += "Логистическая модель не подогнана.\n"

    text_pc1 = (
        f"BMD (5% inc PC1) = {bmd_5pct:.4f} mg/kg\n"
        f"Best PC1 model: {best_name}, params={best_params}\n"
    )
    text_emp = "Эмпирический риск PC1-out (двухстандартное отклонение)."

    # Возвращаем результаты
    results = {
        "fig_log": fig_log,
        "text_log": text_log,
        "fig_pc1": fig_pc1,
        "text_pc1": text_pc1,
        "fig_emp": fig_emp,
        "text_emp": text_emp,
        "BMD": bmd_5pct
    }
    return results, None

###############################################################################
# STREAMLIT ПРИЛОЖЕНИЕ
###############################################################################
def main():
    st.title("Анализ токсичности (Web-приложение)")

    st.write("Загрузите Excel-файл, выберите параметры и нажмите 'Запустить анализ'.")

    # Поля ввода
    uploaded_file = st.file_uploader("Выберите Excel-файл (.xlsx)", type=["xlsx"])
    remove_outliers = st.checkbox("Удалять выбросы (PC1)?", value=True)
    outlier_threshold = st.number_input("Порог выбросов (Z-score):", value=2.0)
    groups_text = st.text_input("Группы доз (через запятую, например 50,100):", value="")
    risk_str = st.text_input("Уровень риска (пример: 1e-4):", value="1e-4")

    # Кнопка
    if st.button("Запустить анализ"):
        if uploaded_file is None:
            st.error("Ошибка: Excel-файл не выбран.")
        else:
            # Проверяем риск
            try:
                user_risk_val = float(risk_str)
            except:
                st.error("Неверное значение риска. Пример: 1e-4")
                return

            with st.spinner("Идёт анализ..."):
                # Вызываем run_analysis
                result, err_msg = run_analysis(
                    file=uploaded_file,
                    remove_outliers=remove_outliers,
                    outlier_threshold=outlier_threshold,
                    groups_text=groups_text,
                    risk_input=user_risk_val
                )

            if err_msg:
                st.error(f"Ошибка: {err_msg}")
            else:
                # Выводим результаты
                st.subheader("Результаты")

                # 1) Логистическая модель
                st.write("### Adjusted Logistic Risk Model")
                st.pyplot(result["fig_log"])
                st.text(result["text_log"])

                # 2) PC1 - Dose
                st.write("### PC1 - Dose")
                st.pyplot(result["fig_pc1"])
                st.text(result["text_pc1"])

                # 3) Empirical Risk
                st.write("### Empirical Risk")
                st.pyplot(result["fig_emp"])
                st.text(result["text_emp"])

                bmd_val = result["BMD"]
                st.success(f"BMD (5% inc PC1) = {bmd_val:.4f} mg/kg/day")


if __name__ == "__main__":
    main()
