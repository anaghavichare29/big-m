import streamlit as st
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

M = 1e6  # Big M value


# ---------- PARSER ---------- #
def parse_objective(obj_str):
    terms = re.findall(r'([+-]?\d*)x(\d+)', obj_str.replace(" ", ""))
    max_var = max(int(var) for _, var in terms)

    c = [0] * max_var
    for coef, var in terms:
        if coef in ["", "+"]:
            coef = 1
        elif coef == "-":
            coef = -1
        else:
            coef = int(coef)
        c[int(var) - 1] = coef

    return c


def parse_constraint(constraint, num_vars):
    parts = re.split(r'(<=|>=|=)', constraint)
    expr, sign, val = parts[0], parts[1], float(parts[2])

    row = [0] * num_vars
    terms = re.findall(r'([+-]?\d*)x(\d+)', expr.replace(" ", ""))

    for coef, var in terms:
        if coef in ["", "+"]:
            coef = 1
        elif coef == "-":
            coef = -1
        else:
            coef = int(coef)

        row[int(var) - 1] = coef

    return row, val, sign


# ---------- BIG M METHOD ---------- #
def big_m_method(c, A, b, signs):
    m = len(A)
    n = len(c)

    table = []
    basis = []
    cb = []

    var_names = [f"x{i+1}" for i in range(n)]
    slack_names = [f"s{i+1}" for i in range(m)]
    artificial_names = [f"A{i+1}" for i in range(m)]

    for i in range(m):
        row = list(A[i])
        slack = [0]*m
        artificial = [0]*m

        if signs[i] == "<=":
            slack[i] = 1
            basis.append(slack_names[i])
            cb.append(0)

        elif signs[i] == ">=":
            slack[i] = -1
            artificial[i] = 1
            basis.append(artificial_names[i])
            cb.append(-M)

        elif signs[i] == "=":
            artificial[i] = 1
            basis.append(artificial_names[i])
            cb.append(-M)

        row += slack + artificial + [b[i]]
        table.append(row)

    cj = c + [0]*m + [-M]*m

    table = np.array(table, dtype=float)
    headers = var_names + slack_names + artificial_names + ["RHS"]

    steps = []

    while True:
        df = pd.DataFrame(table, columns=headers)
        df.insert(0, "Basis", basis)
        df.insert(1, "Cb", cb)

        zj = np.dot(cb, table[:, :-1])
        cj_zj = np.array(cj) - zj

        df.loc["Zj"] = ["", "Zj"] + list(zj) + [""]
        df.loc["Cj-Zj"] = ["", "Cj-Zj"] + list(cj_zj) + [""]

        pivot_col = np.argmax(cj_zj)

        if cj_zj[pivot_col] <= 0:
            steps.append((df.copy(), None, None))
            break

        ratios = []
        for i in range(m):
            if table[i][pivot_col] > 0:
                ratios.append(table[i][-1] / table[i][pivot_col])
            else:
                ratios.append(np.inf)

        pivot_row = np.argmin(ratios)

        df["Ratio"] = ratios + ["", ""]

        steps.append((df.copy(), pivot_row, pivot_col))

        # Pivot operation
        pivot = table[pivot_row][pivot_col]
        table[pivot_row] /= pivot

        for i in range(m):
            if i != pivot_row:
                table[i] -= table[i][pivot_col] * table[pivot_row]

        basis[pivot_row] = headers[pivot_col]
        cb[pivot_row] = cj[pivot_col]

    return steps, table, basis


# ---------- GRAPH ---------- #
import time

def plot_graph_dynamic(A, b, solution, placeholder):
    x = np.linspace(0, 10, 200)

    fig, ax = plt.subplots()

    # Draw constraints one by one (animation feel)
    for i in range(len(A)):
        a1, a2 = A[i]
        if a2 != 0:
            y = (b[i] - a1 * x) / a2

            ax.plot(x, y)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.grid()

            placeholder.pyplot(fig)
            time.sleep(0.6)  # animation delay

    # Plot optimal point at end
    ax.scatter(solution[0], solution[1])
    ax.text(solution[0], solution[1], "Optimal")

    placeholder.pyplot(fig)


# ---------- UI ---------- #
st.title("📊 Big M Method Solver (Notebook Style)")

obj_input = st.text_input("Enter Objective Function", "5x1 + 4x2")

constraints_input = st.text_area(
    "Enter Constraints (one per line)",
    "6x1 + 4x2 >= 24\nx1 + 2x2 = 6\n-x1 + x2 <= 1"
)

if st.button("Solve"):
    try:
        c = parse_objective(obj_input)
        num_vars = len(c)

        A, b, signs = [], [], []

        for line in constraints_input.strip().split("\n"):
            if line.strip() == "":
                continue
            row, val, sign = parse_constraint(line, num_vars)
            A.append(row)
            b.append(val)
            signs.append(sign)

        A_np = np.array(A)
        b_np = np.array(b)

        # Solve using Big M
        steps, final_table, basis = big_m_method(c, A, b, signs)

        st.subheader("📊 Iteration Tables (Big M Method)")

        for i, (df, prow, pcol) in enumerate(steps):
            st.write(f"### Iteration {i}")

            if pcol is not None:
                st.write(f"Pivot Column: {df.columns[pcol+2]}")
                st.write(f"Pivot Row: {prow}")

            st.dataframe(df)

        # Extract solution
        solution = np.zeros(len(c))
        for i, var in enumerate(basis):
            if var.startswith("x"):
                idx = int(var[1:]) - 1
                solution[idx] = final_table[i][-1]

        optimal_value = sum(c[i]*solution[i] for i in range(len(c)))

        st.subheader("✅ Final Result")

        # Show each variable clearly
        for i, val in enumerate(solution):
            st.write(f"x{i+1} = {round(val, 4)}")

        st.write("Optimal Value (Z) =", round(optimal_value, 4))

        # Graph
        if len(c) == 2:            
            st.subheader("📈 Graph")
            graph_placeholder = st.empty()
            plot_graph_dynamic(A, b, solution, graph_placeholder)

    except Exception as e:
        st.error(f"Error: {e}")