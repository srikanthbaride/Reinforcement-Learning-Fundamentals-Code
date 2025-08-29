def approx(x, y, tol=1e-6): 
    return abs(x - y) < tol

def main():
    ok = True
    g0 = 1 + 0.9*2 + (0.9**2)*3
    print("G0 =", g0); ok &= approx(g0, 5.23)
    v = -1 + 0.9*(-1) + (0.9**2)*(-1) + (0.9**3)*10
    print("v =", v); ok &= approx(v, 4.58)
    v_pe = 2 + 0.9*4
    print("v_pe =", v_pe); ok &= approx(v_pe, 5.6, 1e-12)
    q_pe = 1 + 0.9*3
    print("q_pe =", q_pe); ok &= approx(q_pe, 3.7, 1e-12)
    vopt = max(2 + 0.9*5, 1 + 0.9*8)
    print("v* =", vopt); ok &= approx(vopt, 8.2, 1e-12)
    qopt = 2 + 0.9*6
    print("q* =", qopt); ok &= approx(qopt, 7.4, 1e-12)
    v4 = sum((0.9**k)*(-1) for k in range(4)) + (0.9**4)*10
    print("v*(4 steps) =", v4); ok &= abs(v4 - 3.122) < 1e-3
    print("\nALL NUMERIC EXAMPLES:", "PASS" if ok else "FAIL")
    if not ok: raise SystemExit(1)

if __name__ == "__main__":
    main()
