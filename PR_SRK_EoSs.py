import numpy.polynomial.polynomial as poly
import math
import numpy as np


def Volume_SRK(T, P, Tc, Pc, w, R, mol_fractions, Components):
    """Calculates the molecular Volume [M^3/mol] with the Soave-Redlich-Kwong EoS.
        Parameters: P... Pressure Pa
                    T... Temperature in K
                    Tc... critical temperature in K (numpy array with all critical T)
                    Pc... critical pressure in Pa (again in numpy array)
                    w... accentric factor of components (numpy array)
                    mol_fractions... molar fractions of components (array)
                    R... molar gasconstant (call C_R())
                    Components... array of arrays (for each component 1) of tuples of groups + how often group appears in Component
                                  if component is only consisting of one group add array of 1 element [make sure Groups are ordered after ID]
                        groups: 0... CH3
                                1... CH2
                                2... CH4
                                3... C2H6
                                4... CO2
                                5... N2
                                6... H2
                                7... CH
        Implemented with help of Giorgio Soave 1971 [https://doi.org/10.1016/0009-2509(72)80096-4]"""
    Num_Components = len(Tc)
    ai = np.zeros(Num_Components)
    bi = np.zeros(Num_Components)
    v = []

    Omega_a = 0.42747
    Omega_b = 0.08664

    # calculate a and b
    for i in range(Num_Components):
        bi[i] = Omega_b * R * Tc[i] / Pc[i]
        mi = 0.480 + 1.574 * w[i] - 0.17 * math.pow(w[i], 2)
        ai[i] = Omega_a * math.pow(R * Tc[i] * (1 + mi * (1 - math.sqrt(T / Tc[i]))), 2) / Pc[i]

    # use mixing rule if more than one component
    if Num_Components > 1:
        a, b = CMR_MixingRule(T, mol_fractions, ai, bi, Components)
    else:
        a = ai[0]
        b = bi[0]

    p0 = -a * b
    p1 = a - R * T * b - P * math.pow(b, 2)
    p2 = -R * T
    p3 = P

    Po = poly.Polynomial((p0, p1, p2, p3))
    for root in Po.roots():
        if np.imag(root) == 0.:
            v.append(float(root) * 1000000)

    return v[-1]

def Volume_PR(T, P, Tc, Pc, w, R, mol_fractions, Components):
    """Calculates the molecular Volume [M^3/mol] with the Soave-Redlich-Kwong EoS.
        Parameters: P... Pressure Pa
                    T... Temperature in K
                    Tc... critical temperature in K (numpy array with all critical T)
                    Pc... critical pressure in Pa (again in numpy array)
                    w... accentric factor of components (numpy array)
                    mol_fractions... molar fractions of components (array)
                    R... molar gasconstant (call C_R())
                    Components... array of arrays (for each component 1) of tuples of groups + how often group appears in Component
                                  if component is only consisting of one group add array of 1 element [make sure Groups are ordered after ID]
                        groups: 0... CH3
                                1... CH2
                                2... CH4
                                3... C2H6
                                4... CO2
                                5... N2
                                6... H2
                                7... CH
        Implemented with help of Jun-Wei Qian 2012 [http://dx.doi.org/10.1016/j.supflu.2012.12.014]"""
    Num_Components = len(Tc)
    ai = np.zeros(Num_Components)
    bi = np.zeros(Num_Components)
    v = []

    Omega_a = C_Omega_a()
    Omega_b = C_Omega_b()

    # calculate a and b
    for i in range(Num_Components):
        bi[i] = Omega_b * R * Tc[i] / Pc[i]
        if w[i] > 0.491:
            mi = 0.379642 + 1.48503 * w[i] - 0.164423 * math.pow(w[i], 2) + 0.016666 * math.pow(w[i], 3)
        else:
            mi = 0.37464 + 1.54226 * w[i] - 0.26992 * math.pow(w[i], 2)
        ai[i] = Omega_a * math.pow(R * Tc[i] * (1 + mi * (1 - math.sqrt(T / Tc[i]))), 2) / Pc[i]

    # use mixing rule if more than one component
    if Num_Components > 1:
        a, b = VanDerWaals_MixingRule(T, mol_fractions, ai, bi, Components)
    else:
        a = ai[0]
        b = bi[0]

    p0 = -a * b + R * T * math.pow(b, 2) + P * math.pow(b, 3)
    p1 = a - 2 * R * T * b - 3 * P * math.pow(b, 2)
    p2 = P * b - R * T
    p3 = P

    Po = poly.Polynomial((p0, p1, p2, p3))
    for root in Po.roots():
        if np.imag(root) == 0.:
            v.append(float(root) * 1000000)

    return v[-1]

def Pressure_Peng_Robinson(v, T, Tc, Pc, w, R, mol_fractions, Components):
    """Calculates the Pressure with the Peng_Robinson EoS.
        Parameters: v... molecular Volume in M3/mol
                    T... Temperature in K
                    Tc... critical temperature in K (numpy array with all critical T)
                    Pc... critical pressure in Pa (again in numpy array)
                    w... accentric factor of components (numpy array)
                    mol_fractions... molar fractions of components (array)
                    R... molar gasconstant (call C_R())
                    Components... array of arrays (for each component 1) of tuples of groups + how often group appears in Component
                                  if component is only consisting of one group add array of 1 element [make sure Groups are ordered after ID]
                        groups: 0... CH3
                                1... CH2
                                2... CH4
                                3... C2H6
                                4... CO2
                                5... N2
                                6... H2
                                7... CH
        Implemented with help of Jun-Wei Qian 2012 [http://dx.doi.org/10.1016/j.supflu.2012.12.014]"""
    Num_Components = len(Tc)
    ai = np.zeros(Num_Components)
    bi = np.zeros(Num_Components)

    Omega_a = C_Omega_a()
    Omega_b = C_Omega_b()

    for i in range(Num_Components):
        bi[i] = Omega_b * R * Tc[i] / Pc[i]
        if w[i] > 0.491:
            mi = 0.379642 + 1.48503 * w[i] - 0.164423 * math.pow(w[i], 2) + 0.016666 * math.pow(w[i], 3)
        else:
            mi = 0.37464 + 1.54226 * w[i] - 0.26992 * math.pow(w[i], 2)
        ai[i] = Omega_a * math.pow(R * Tc[i] * (1 + mi * (1 - math.sqrt(T / Tc[i]))), 2) / Pc[i]

    if Num_Components > 1:
        a, b = VanDerWaals_MixingRule(T, mol_fractions, ai, bi, Components)
    else:
        a = ai[0]
        b = bi[0]

    return R * T / (v - b) - a / (v * (v + b) + b * (v - b))

def Pressure_SRK(v, T, Tc, Pc, w, R, mol_fractions, Components):
    """Calculates the Pressure with the Peng_Robinson EoS.
        Parameters: v... molecular Volume in M3/mol
                    T... Temperature in K
                    Tc... critical temperature in K (numpy array with all critical T)
                    Pc... critical pressure in Pa (again in numpy array)
                    w... accentric factor of components (numpy array)
                    mol_fractions... molar fractions of components (array)
                    R... molar gasconstant (call C_R())
                    Components... array of arrays (for each component 1) of tuples of groups + how often group appears in Component
                                  if component is only consisting of one group add array of 1 element [make sure Groups are ordered after ID]
                        groups: 0... CH3
                                1... CH2
                                2... CH4
                                3... C2H6
                                4... CO2
                                5... N2
                                6... H2
                                7... CH
                                Implemented with help of Giorgio Soave 1971 [https://doi.org/10.1016/0009-2509(72)80096-4] """
    Num_Components = len(Tc)
    ai = np.zeros(Num_Components)
    bi = np.zeros(Num_Components)

    Omega_a = 0.42747
    Omega_b = 0.08664

    for i in range(Num_Components):
        bi[i] = Omega_b * R * Tc[i] / Pc[i]
        mi = 0.480 + 1.574 * w[i] - 0.17 * math.pow(w[i], 2)
        ai[i] = Omega_a * math.pow(R * Tc[i] * (1 + mi * (1 - math.sqrt(T / Tc[i]))), 2) / Pc[i]

    if Num_Components > 1:
        a, b = CMR_MixingRule(T, mol_fractions, ai, bi, Components)
    else:
        a = ai[0]
        b = bi[0]

    return R * T / (v - b) - a / (v * (v + b))


def C_R_NIST():
    """Molecular gas constant R as measured by the NIST https://physics.nist.gov/cgi-bin/cuu/Value?r|search_for=gas+constant"""
    return 8.314462618


def C_R():
    """Molecular gas constant as given by paper Jun-Wei Qian 2012 [http://dx.doi.org/10.1016/j.supflu.2012.12.014]"""
    return 8.314472