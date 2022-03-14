from numpy.testing import assert_allclose
from pylira.utils.priors import f_hyperprior_esch, f_hyperprior_lira


def test_f_hyperprior_esch():
    value = f_hyperprior_esch(alpha=0.2, delta=3000, index=0, index_alpha=3)
    assert_allclose(value, 0.000335, atol=1e-5)


def test_f_hyperprior_lira():
    value = f_hyperprior_lira(alpha=0.2, ms_al_kap1=0, ms_al_kap2=1000, ms_al_kap3=3)
    assert_allclose(value, 0.000335, atol=1e-5)
