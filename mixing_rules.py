def VanDerWaals_MixingRule(T, z, ai, bi, Components):
    ab = np.zeros(2)
    A = VW_Akl_PR()
    B = VW_Bkl_PR()

    kij = VW_kij(ai, bi, Components, A, B, T)

    for i in range(len(ai)):
        ab[1] += z[i] * bi[i]
        for j in range(len(ai)):
            ab[0] += z[i] * z[j] * math.sqrt(ai[i] * ai[j]) * (1 - kij[i, j])
    return ab[0], ab[1]

def VW_kij(ai, bi, Components, A, B, T):
    """implemented with help of Jean-Noël Jaubert 2004 [https://doi.org/10.1016/j.fluid.2004.06.059]"""
    N = len(ai)
    k = np.zeros((N, N))

    for i, ith_Comp in enumerate(Components):
        for j, jth_Comp in enumerate(Components):
            if i == j:
                k[i, j] = 0
            else:
                div = 2 * math.sqrt(ai[i] * ai[j]) / (bi[i] * bi[j])
                k[i, j] = (-1 / 2 * VW_group_sum_kij(ith_Comp, jth_Comp, A, B, T) - np.power(
                    math.sqrt(ai[i]) / bi[i] - math.sqrt(ai[j]) / bi[j], 2)) / div
    return k

def VW_group_sum_kij(ith_Component, jth_Component, A, B, T):
    """calculate the double sum in kij for the Van der Waals mixing rule"""

    summ = 0

    Num_Groups_i = 0
    Groups_i = []
    alpha_i = []
    Num_Groups_j = 0
    Groups_j = []
    alpha_j = []
    range_groups = []

    for group in ith_Component:
        Num_Groups_i += group[1]  # calculate total num of groups in ith comp
        Groups_i.append(group[0])

    range_groups = Groups_i.copy()

    for group in jth_Component:
        Num_Groups_j += group[1]
        Groups_j.append(group[0])
        if group[0] in range_groups:  # only add new ones
            pass
        else:
            range_groups.append(group[0])

    range_groups.sort()  # only range_groups interesting all óther sum elements are 0

    for k in range_groups:
        if k in Groups_i:
            ai = ith_Component[Groups_i.index(k)][1] / Num_Groups_i  # number of component in i/num of total comp in i
        else:
            ai = 0
        if k in Groups_j:
            aj = jth_Component[Groups_j.index(k)][1] / Num_Groups_j
        else:
            aj = 0

        alpha_i.append(ai)
        alpha_j.append(aj)

    for ind_k, k in enumerate(range_groups):
        for ind_l, l in enumerate(range_groups):
            a_a = (alpha_i[ind_k] - alpha_j[ind_k]) * (alpha_i[ind_l] - alpha_j[ind_l]) * A[k, l]

            if a_a == 0:
                summ += 0

            else:
                summ += a_a * np.power((298.15 / T), (B[k, l] / A[
                    k, l] - 1))  # only use if a_a!=0 otherwise A[k,l] could be 0 leading to an error in B/A

    return summ

def VW_Akl_PR():
    """from Jun-Wei Qian 2012 [http://dx.doi.org/10.1016/j.supflu.2012.12.014]"""
    A = np.zeros((8, 8))

    A[0, 1], A[1, 0] = 74.81, 74.81  # in MPa
    A[0, 2], A[2, 0] = 32.94, 32.94
    A[0, 3], A[3, 0] = 8.579, 8.579
    A[0, 4], A[4, 0] = 164, 164
    A[0, 5], A[5, 0] = 52.74, 52.74
    A[0, 6], A[6, 0] = 202.8, 202.8
    A[0, 7], A[7, 0] = 261.5, 261.5

    A[1, 2], A[2, 1] = 36.72, 36.72
    A[1, 3], A[3, 1] = 31.23, 31.23
    A[1, 4], A[4, 1] = 136.9, 136.9
    A[1, 5], A[5, 1] = 82.28, 82.28
    A[1, 6], A[6, 1] = 132.5, 132.5
    A[1, 7], A[7, 1] = 51.47, 51.47

    A[2, 3], A[3, 2] = 13.04, 13.04
    A[2, 4], A[4, 2] = 137.3, 137.3
    A[2, 5], A[5, 2] = 37.9, 37.9
    A[2, 6], A[6, 2] = 156.1, 156.1
    A[2, 7], A[7, 2] = 145.2, 145.2

    A[3, 4], A[4, 3] = 135.5, 135.5
    A[3, 5], A[5, 3] = 61.59, 61.59
    A[3, 6], A[6, 3] = 137.6, 137.6
    A[3, 7], A[7, 3] = 145.2, 145.2

    A[4, 5], A[5, 4] = 98.42, 98.42
    A[4, 6], A[6, 4] = 265.9, 265.9
    A[4, 7], A[7, 4] = 184.3, 184.3

    A[5, 6], A[6, 5] = 65.2, 65.2
    A[5, 7], A[7, 5] = 365.4, 365.4

    A[6, 7], A[7, 6] = 415.2, 415.2

    A = A * 100000  # translate into Pa
    return A


def VW_Bkl_PR():
    "from Jun-Wei Qian 2012 [http://dx.doi.org/10.1016/j.supflu.2012.12.014]"
    B = np.zeros((8, 8))

    B[0, 1], B[1, 0] = 165.7, 165.7
    B[0, 2], B[2, 0] = -35, -35
    B[0, 3], B[3, 0] = -29.51, -29.51
    B[0, 4], B[4, 0] = 269, 269
    B[0, 5], B[5, 0] = 87.19, 87.19
    B[0, 6], B[6, 0] = 317.4, 317.4
    B[0, 7], B[7, 0] = 388.8, 388.8

    B[1, 2], B[2, 1] = 108.4, 108.4
    B[1, 3], B[3, 1] = 84.76, 84.76
    B[1, 4], B[4, 1] = 254.6, 254.6
    B[1, 5], B[5, 1] = 202.8, 202.8
    B[1, 6], B[6, 1] = 147.2, 147.2
    B[1, 7], B[7, 1] = 79.61, 79.61

    B[2, 3], B[3, 2] = 6.863, 6.863
    B[2, 4], B[4, 2] = 194.2, 194.2
    B[2, 5], B[5, 2] = 37.2, 37.2
    B[2, 6], B[6, 2] = 92.99, 92.99
    B[2, 7], B[7, 2] = 301.6, 301.6

    B[3, 4], B[4, 3] = 239.5, 239.5
    B[3, 5], B[5, 3] = 84.92, 84.92
    B[3, 6], B[6, 3] = 150, 150
    B[3, 7], B[7, 3] = 352.1, 352.1

    B[4, 5], B[5, 4] = 221.4, 221.4
    B[4, 6], B[6, 4] = 268.3, 268.3
    B[4, 7], B[7, 4] = 762.1, 762.1

    B[5, 6], B[6, 5] = 70.1, 70.1
    B[5, 7], B[7, 5] = 521.9, 521.9

    B[6, 7], B[7, 6] = 726.4, 726.4

    B = B * 100000
    return B


def C_X():
    return (-1 + math.pow(6 * math.sqrt(2) + 8, 1 / 3) - math.pow(6 * math.sqrt(2) - 8, 1 / 3)) / 3


def C_Omega_b():
    return C_X() / (C_X() + 3)


def C_Omega_a():
    return 8 * (5 * C_X() + 1) / (49 - 37 * C_X())

def CMR_MixingRule(T, z, ai, bi, Components):
    """Classical mixing rule by Andrzej Anderko 2000 in "Equation of states for Fluids and Fluid Mixtures" [https://doi.org/10.1016/S1874-5644(00)80015-6] """
    ab = np.zeros(2)
    A = VW_Akl_SRK()
    B = VW_Bkl_SRK()

    kij = CMR_kij_SRK(ai, bi, Components, A, B, T)

    for i in range(len(ai)):
        ab[1] += z[i] * bi[i]
        for j in range(len(ai)):
            ab[0] += z[i] * z[j] * math.sqrt(ai[i] * ai[j]) * (1 - kij[i, j])

    return ab[0], ab[1]


def CMR_kij_SRK(ai, bi, Components, A, B, T):
    """implemented with help of Jean-Noël Jaubert 2004 [https://doi.org/10.1016/j.fluid.2004.06.059]"""
    N = len(ai)
    k = np.zeros((N, N))

    for i, ith_Comp in enumerate(Components):
        for j, jth_Comp in enumerate(Components):
            if i == j:
                k[i, j] = 0
            else:
                div = 2 * math.sqrt(ai[i] * ai[j]) / (bi[i] * bi[j])
                k[i, j] = (-1 / 2 * VW_group_sum_kij(ith_Comp, jth_Comp, A, B, T) - np.power(
                    math.sqrt(ai[i]) / bi[i] - math.sqrt(ai[j]) / bi[j], 2)) / div
    return k


def VW_Akl_SRK():
    """from Jean-Noël Jaubert 2010 [doi:10.1016/j.fluid.2010.03.037]"""
    A = np.zeros((8, 8))

    A[0, 1], A[1, 0] = 74.81, 74.81  # in MPa
    A[0, 2], A[2, 0] = 32.94, 32.94
    A[0, 3], A[3, 0] = 8.579, 8.579
    A[0, 4], A[4, 0] = 164, 164
    A[0, 5], A[5, 0] = 52.74, 52.74
    A[0, 6], A[6, 0] = 202.8, 202.8
    A[0, 7], A[7, 0] = 261.5, 261.5

    A[1, 2], A[2, 1] = 36.72, 36.72
    A[1, 3], A[3, 1] = 31.23, 31.23
    A[1, 4], A[4, 1] = 136.9, 136.9
    A[1, 5], A[5, 1] = 82.28, 82.28
    A[1, 6], A[6, 1] = 132.5, 132.5
    A[1, 7], A[7, 1] = 51.47, 51.47

    A[2, 3], A[3, 2] = 13.04, 13.04
    A[2, 4], A[4, 2] = 137.3, 137.3
    A[2, 5], A[5, 2] = 37.9, 37.9
    A[2, 6], A[6, 2] = 156.1, 156.1
    A[2, 7], A[7, 2] = 145.2, 145.2

    A[3, 4], A[4, 3] = 135.5, 135.5
    A[3, 5], A[5, 3] = 61.59, 61.59
    A[3, 6], A[6, 3] = 137.6, 137.6
    A[3, 7], A[7, 3] = 145.2, 145.2

    A[4, 5], A[5, 4] = 98.42, 98.42
    A[4, 6], A[6, 4] = 265.9, 265.9
    A[4, 7], A[7, 4] = 184.3, 184.3

    A[5, 6], A[6, 5] = 65.2, 65.2
    A[5, 7], A[7, 5] = 365.4, 365.4

    A[6, 7], A[7, 6] = 415.2, 415.2

    A = A * 100000 * 0.808  # translate into Pa
    return A


def VW_Bkl_SRK():
    "from Jean-Noël Jaubert 2010 [doi:10.1016/j.fluid.2010.03.037]"
    B = np.zeros((8, 8))

    B[0, 1], B[1, 0] = 165.7, 165.7
    B[0, 2], B[2, 0] = -35, -35
    B[0, 3], B[3, 0] = -29.51, -29.51
    B[0, 4], B[4, 0] = 269, 269
    B[0, 5], B[5, 0] = 87.19, 87.19
    B[0, 6], B[6, 0] = 317.4, 317.4
    B[0, 7], B[7, 0] = 388.8, 388.8

    B[1, 2], B[2, 1] = 108.4, 108.4
    B[1, 3], B[3, 1] = 84.76, 84.76
    B[1, 4], B[4, 1] = 254.6, 254.6
    B[1, 5], B[5, 1] = 202.8, 202.8
    B[1, 6], B[6, 1] = 147.2, 147.2
    B[1, 7], B[7, 1] = 79.61, 79.61

    B[2, 3], B[3, 2] = 6.863, 6.863
    B[2, 4], B[4, 2] = 194.2, 194.2
    B[2, 5], B[5, 2] = 37.2, 37.2
    B[2, 6], B[6, 2] = 92.99, 92.99
    B[2, 7], B[7, 2] = 301.6, 301.6

    B[3, 4], B[4, 3] = 239.5, 239.5
    B[3, 5], B[5, 3] = 84.92, 84.92
    B[3, 6], B[6, 3] = 150, 150
    B[3, 7], B[7, 3] = 352.1, 352.1

    B[4, 5], B[5, 4] = 221.4, 221.4
    B[4, 6], B[6, 4] = 268.3, 268.3
    B[4, 7], B[7, 4] = 762.1, 762.1

    B[5, 6], B[6, 5] = 70.1, 70.1
    B[5, 7], B[7, 5] = 521.9, 521.9

    B[6, 7], B[7, 6] = 726.4, 726.4

    B = B * 100000 * 0.808
    return B