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
# ФУНКЦИИ ДЛЯ АНАЛИЗА
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
# МОДЕЛИ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
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
        popt, _ = curve_fit(model_func, dose, pc1, p0=p0, maxfev=20000)  # Уменьшили maxfev
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
# ОСНОВНАЯ ФУНКЦИЯ АНАЛИЗА
###############################################################################
def run_analysis(
    file,
    remove_outliers,
    outlier_threshold,
    groups_text,
    risk_input,
    normalize_control,
    normalize_experimental,
    slope_threshold=5
):
    # 1) Читаем Excel
    df = pd.read_excel(file)

    # 2) Преобразуем все столбцы в числа
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3) Транслитерация
    old_cols = df.columns.tolist()
    new_cols = [transliterate_column_name(col) for col in old_cols]
    df.columns = new_cols

    # 4) Ищем столбец "Dose"
    dose_col = None
    for c in df.columns:
        if c.lower() in ["doza", "dose", "доза", "доза_"] or c.upper() == "DOZA":
            dose_col = c
            break
    if dose_col is None:
        return None, "Столбец с дозой не найден."

    df.rename(columns={dose_col: "Dose"}, inplace=True)

    # 5) Заполняем пропуски медианами
    all_cols = df.columns.tolist()
    predictors = [col for col in all_cols if col != "Dose"]
    for col in predictors:
        df[col].fillna(df[col].median(), inplace=True)

    # 6) Удаляем столбцы, где нет значений
    grouped = df.groupby("Dose")
    cols_to_drop = []
    for col in predictors:
        if any(grouped[col].count() == 0):
            cols_to_drop.append(col)
    cols_to_drop = list(set(cols_to_drop))
    df.drop(columns=cols_to_drop, inplace=True)
    final_predictors = [col for col in predictors if col not in cols_to_drop]

    # 7) PCA -> PC1
    X = df[final_predictors].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    df["PC1"] = X_pca[:, 0]

    # 8) Нормализация контрольной группы (Dose==0)
    if normalize_control:
        control_mask = (df["Dose"] == 0)
        if control_mask.sum() > 0:
            positive_mask = (df["Dose"] > 0)
            if positive_mask.sum() > 0:
                min_positive = df.loc[positive_mask, "Dose"].min()
                group_min = df[df["Dose"] == min_positive]
                mean_min = group_min["PC1"].mean()
                while True:
                    group0 = df[df["Dose"] == 0]
                    if len(group0) <= 1:
                        break
                    mean_control = group0["PC1"].mean()
                    if mean_control <= mean_min:
                        break
                    idx_max = group0["PC1"].idxmax()
                    valid_control = group0[group0["PC1"] < mean_min]["PC1"]
                    if len(valid_control) == 0:
                        break
                    sample_valid = valid_control.sample(frac=0.5, random_state=42)
                    new_value = sample_valid.mean()
                    df.loc[idx_max, "PC1"] = new_value

    # 9) Нормализация опытных групп
    if normalize_experimental:
        st.write("=== Начинаем нормализацию опытных групп ===")
        positive_doses = sorted(df.loc[df["Dose"] > 0, "Dose"].unique())

        # 9.1. Базовая корректировка
        st.write("Шаг 9.1: Базовая корректировка (среднее в группе не должно быть меньше предыдущей).")
        for i in range(1, len(positive_doses)):
            current_dose = positive_doses[i]
            prev_dose = positive_doses[i - 1]
            group_current = df[df["Dose"] == current_dose]
            group_prev = df[df["Dose"] == prev_dose]
            mean_current = group_current["PC1"].mean()
            mean_prev = group_prev["PC1"].mean()
            st.write(f"   Доза={current_dose}, среднее={mean_current:.4f}, предыдущая доза={prev_dose}, среднее={mean_prev:.4f}")
            while mean_current < mean_prev:
                idx_min = group_current["PC1"].idxmin()
                sample_prev = group_prev["PC1"].sample(frac=0.5, random_state=42)
                if sample_prev.empty:
                    st.write("      Не осталось значений для выборки; прерываем цикл.")
                    break
                new_value = sample_prev.mean()
                st.write(f"      Заменяем минимальное значение в группе {current_dose} на {new_value:.4f}")
                df.loc[idx_min, "PC1"] = new_value
                group_current = df[df["Dose"] == current_dose]
                mean_current = group_current["PC1"].mean()
                st.write(f"      Новое среднее в группе {current_dose}={mean_current:.4f}")

        # 9.2. Дополнительная корректировка при крутом наклоне - максимум 3 итерации
        tries = 0
        while tries < 3:
            tries += 1
            st.write(f"--- Доп. корректировка №{tries}, проверяем наклон логистики. ---")

            # Считаем «временную» логистику
            try:
                df_control_temp = df[df["Dose"] == 0]
                if len(df_control_temp) == 0:
                    mu_c_temp = df["PC1"].mean()
                    sd_c_temp = df["PC1"].std()
                else:
                    mu_c_temp = df_control_temp["PC1"].mean()
                    sd_c_temp = df_control_temp["PC1"].std()

                low_bound_temp = mu_c_temp - 2*sd_c_temp
                up_bound_temp = mu_c_temp + 2*sd_c_temp
                df["PC1_out_temp"] = ~df["PC1"].between(low_bound_temp, up_bound_temp)
                risk_df_temp = df.groupby("Dose")["PC1_out_temp"].mean()
                doses_risk_temp = np.array(sorted(risk_df_temp.index))
                risk_vals_emp_temp = np.array([risk_df_temp.loc[d] for d in doses_risk_temp])

                popt_log_temp, _ = curve_fit(
                    risk_logistic_adj, 
                    doses_risk_temp, 
                    risk_vals_emp_temp,
                    p0=[50, 0.1], 
                    maxfev=20000  # ограниченное число итераций
                )
            except RuntimeError:
                popt_log_temp = None

            if popt_log_temp is None:
                st.write("Не удалось подогнать логистическую кривую (RuntimeError). Прерываем доп. корректировку.")
                break

            # popt_log_temp = [A_temp, B_temp]
            A_temp, B_temp = popt_log_temp
            st.write(f"Параметры временной логистики: A_temp={A_temp:.5f}, B_temp={B_temp:.5f}")
            if B_temp <= slope_threshold:
                st.write(f"Наклон B={B_temp:.5f} ниже порога {slope_threshold}, корректировка не требуется.")
                break
            else:
                st.write(f"Наклон B={B_temp:.5f} все еще выше порога {slope_threshold} -> ищем резкий подъем.")
                # Находим группу с max приростом (mean_curr - mean_prev)
                sharp_group = None
                sharp_diff = -np.inf
                for i_pd in range(1, len(positive_doses)):
                    curr_dose = positive_doses[i_pd]
                    prev_dose = positive_doses[i_pd - 1]
                    group_curr = df[df["Dose"] == curr_dose]
                    group_prev = df[df["Dose"] == prev_dose]
                    diff = group_curr["PC1"].mean() - group_prev["PC1"].mean()
                    if diff > sharp_diff:
                        sharp_diff = diff
                        sharp_group = curr_dose
                st.write(f"Резкий подъем найден в группе {sharp_group} (прирост={sharp_diff:.4f}).")

                # Проверяем, есть ли у этой группы предыдущая и следующая
                if sharp_group is not None:
                    sharp_index = positive_doses.index(sharp_group)
                    if sharp_index > 0 and sharp_index < len(positive_doses) - 1:
                        group_prev = df[df["Dose"] == positive_doses[sharp_index - 1]]
                        group_next = df[df["Dose"] == positive_doses[sharp_index + 1]]
                        group_sharp = df[df["Dose"] == sharp_group]
                        count_sharp = len(group_sharp)
                        st.write(f"Группа {sharp_group}, кол-во значений={count_sharp}. Меняем каждое.")
                        for idx in group_sharp.index:
                            combined_values = pd.concat([group_prev["PC1"], group_next["PC1"]])
                            if len(combined_values) == 0:
                                st.write("Нет значений для объединения; прерываем.")
                                break
                            sample_combined = combined_values.sample(frac=0.5, random_state=42)
                            if not sample_combined.empty:
                                new_value = sample_combined.mean()
                                st.write(f"   Заменяем значение в {sharp_group}, idx={idx}, на {new_value:.4f}")
                                df.loc[idx, "PC1"] = new_value
                    else:
                        st.write("У этой группы нет предыдущей/следующей (или она крайняя). Прерываем корректировку.")
                        break

                # если sharp_group=None или что-то не сработало – мы всё равно идём на следующий tries
                # если на следующем шаге B всё равно высокий – повторим
        st.write("=== Завершили нормализацию опытных групп ===")

    # 10) Удаляем выбросы
    if remove_outliers:
        if groups_text.strip() == "":
            groups_to_clean = None
        else:
            try:
                groups_to_clean = [float(x.strip()) for x in groups_text.split(",")]
            except:
                return None, "Неверный формат групп доз (пример: 50,100)."
        df = replace_outliers_in_PC1(
            df, group_col="Dose", col="PC1",
            threshold=outlier_threshold,
            groups_to_clean=groups_to_clean
        )

    # 11) Подбираем лучшую модель (Linear, Quadratic, Log, Exponential, Power)
    dose_vals = df["Dose"].values
    pc1_vals = df["PC1"].values
    model_candidates = [
        ("Linear", model_linear, [0, 0]),
        ("Quadratic", model_quadratic, [0, 0, 0]),
        ("Log", model_log, [0, 0]),
        ("Exponential", model_exp, [0, 1, 0]),
        ("Power", model_power, [0, 1, 1])
    ]
    results = []
    for (mname, mfunc, p0) in model_candidates:
        r_ = fit_and_evaluate(mfunc, dose_vals, pc1_vals, p0)
        results.append((mname, r_))
    results_sorted = sorted(results, key=lambda x: x[1]["AIC"])
    best_name, best_res = results_sorted[0]
    best_func = [x for x in model_candidates if x[0] == best_name][0][1]
    best_params = best_res["params"]

    # 12) График PC1 ~ Dose
    fig_pc1, ax1 = plt.subplots()
    ax1.scatter(dose_vals, pc1_vals, alpha=0.6, label="Data PC1")
    dose_grid = np.linspace(dose_vals.min(), dose_vals.max(), 200)
    pc1_pred = best_func(dose_grid, *best_params)
    ax1.plot(dose_grid, pc1_pred, 'r--', label=f"{best_name} fit")
    ax1.set_xlabel("Dose")
    ax1.set_ylabel("PC1")
    ax1.set_title("PC1 ~ Dose")
    ax1.legend()

    # 13) BMD (5% прирост от PC1(0))
    pc1_0 = best_func(0, *best_params)
    target_pc1 = pc1_0 * 1.05
    bmd_5pct = find_dose_for_target(
        target_pc1, 
        best_func, 
        best_params,
        d_min=0, 
        d_max=1000, 
        step=0.1
    )

    # 14) Эмпирический риск (двухстандартное отклонение от контроля)
    df_control = df[df["Dose"] == 0]
    if len(df_control) == 0:
        mu_c = df["PC1"].mean()
        sd_c = df["PC1"].std()
    else:
        mu_c = df_control["PC1"].mean()
        sd_c = df_control["PC1"].std()
    low_bound = mu_c - 2 * sd_c
    up_bound = mu_c + 2 * sd_c
    df["PC1_out"] = ~df["PC1"].between(low_bound, up_bound)
    risk_df = df.groupby("Dose")["PC1_out"].mean()
    if 0 in risk_df.index:
        risk_df.loc[0] = 0.0

    fig_emp, ax2 = plt.subplots()
    ax2.plot(risk_df.index, risk_df.values * 100, 'ro-')
    ax2.set_xlabel("Dose")
    ax2.set_ylabel("Risk (%)")
    ax2.set_title("Empirical Risk")

    # 15) Логистическая модель риска
    try:
        doses_risk = np.array(sorted(risk_df.index))
        risk_vals_emp = np.array([risk_df.loc[d] for d in doses_risk])
        popt_log, _ = curve_fit(
            risk_logistic_adj, 
            doses_risk, 
            risk_vals_emp,
            p0=[50, 0.1], 
            maxfev=20000
        )
        fig_log, ax3 = plt.subplots()
        d_grid_log = np.linspace(doses_risk.min(), doses_risk.max() * 1.2, 150)
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
        ax_.text(0.1, 0.5, "Logistic model fitting failed.", fontsize=12)
        ax_.set_title("Adjusted Logistic Risk Model")

    # 16) Формируем текстовые результаты
    text_log = "Скорректированная логистическая функция:\n"
    text_log += "r(d) = [ f(d) - f(0) ] / [ 1 - f(0) ], где f(d) = 1/[1 + exp(-B*(d - A))].\n\n"

    if popt_log is not None:
        A_est, B_est = popt_log
        text_log += f"Параметры логистической модели: A={A_est:.5f}, B={B_est:.5f}\n"
        user_risk_dose = find_dose_for_risk_adj(risk_input, A_est, B_est)
        text_log += f"\nВведённый риск: {risk_input:.5f} => Dose={user_risk_dose:.5f}\n\n"
        standard_risks = [1e-3, 1e-4, 1e-5]
        for rr in standard_risks:
            dd_ = find_dose_for_risk_adj(rr, A_est, B_est)
            r_str = format_sci_custom(rr, precision=1)
            text_log += f"Risk={r_str} => Dose={dd_:.5f} mg/kg/day\n"
    else:
        text_log += "Логистическая модель не подогнана.\n"

    text_pc1 = (
        f"BMD (5% inc PC1) = {bmd_5pct:.4f} mg/kg\n"
        f"Best PC1 model: {best_name}, params={best_params}\n"
    )
    text_emp = "Эмпирический риск PC1-out (двухстандартное отклонение)."

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
# STREAMLIT-ПРИЛОЖЕНИЕ
###############################################################################
def main():
    st.title("Анализ токсичности (Web-приложение)")
    st.write("Загрузите Excel-файл, выберите параметры и нажмите 'Запустить анализ'.")

    # Интерфейс ввода
    uploaded_file = st.file_uploader("Выберите Excel-файл (.xlsx)", type=["xlsx"])
    remove_outliers = st.checkbox("Удалять выбросы (PC1)?", value=True)
    outlier_threshold = st.number_input("Порог выбросов (Z-score):", value=2.0)
    groups_text = st.text_input("Группы доз (через запятую, например 50,100):", value="")
    risk_str = st.text_input("Уровень риска (пример: 1e-4):", value="1e-4")
    normalize_control = st.checkbox("Нормализация контрольной группы.", value=False)
    normalize_experimental = st.checkbox("Нормализация опытных групп.", value=False)
    slope_threshold = st.number_input("Порог наклона лог. регрессии:", value=5.0)

    # Кнопка запуска
    if st.button("Запустить анализ"):
        if uploaded_file is None:
            st.error("Ошибка: Excel-файл не выбран.")
        else:
            # Парсим риск
            try:
                user_risk_val = float(risk_str)
            except:
                st.error("Неверное значение риска. Пример: 1e-4")
                return

            with st.spinner("Идёт анализ..."):
                result, err_msg = run_analysis(
                    file=uploaded_file,
                    remove_outliers=remove_outliers,
                    outlier_threshold=outlier_threshold,
                    groups_text=groups_text,
                    risk_input=user_risk_val,
                    normalize_control=normalize_control,
                    normalize_experimental=normalize_experimental,
                    slope_threshold=slope_threshold
                )

            # Обрабатываем результат
            if err_msg:
                st.error(f"Ошибка: {err_msg}")
            else:
                st.subheader("Результаты")

                st.write("### Adjusted Logistic Risk Model")
                st.pyplot(result["fig_log"])
                st.text(result["text_log"])

                st.write("### PC1 - Dose")
                st.pyplot(result["fig_pc1"])
                st.text(result["text_pc1"])

                st.write("### Empirical Risk")
                st.pyplot(result["fig_emp"])
                st.text(result["text_emp"])

                bmd_val = result["BMD"]
                st.success(f"BMD (5% inc PC1) = {bmd_val:.4f} mg/kg/day")


if __name__ == "__main__":
    main()

