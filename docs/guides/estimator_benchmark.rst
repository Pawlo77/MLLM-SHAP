Estimator benchmark
===================

Every built-in estimator on five reference games of ten players: additive, pairwise
interaction, airport, weighted voting and random unanimity. The reference is the
exact value from all 1,024 coalitions. Budgets are :math:`\alpha n` coalitions
(``budget.per_player: α``). Each cell is the mean over the games and 50 seeds of the
relative :math:`\ell_2` error, the top-five overlap, the efficiency gap
:math:`|\sum_i \hat\phi_i - (v(N) - v(\emptyset))| / |v(N) - v(\emptyset)|` and the
spread of the error over seeds. ``examples/estimator_comparison.ipynb`` (on
:doc:`../examples`) regenerates both tables.

Shapley
-------

.. list-table:: Shapley estimand
   :header-rows: 1
   :widths: 24 8 16 16 16 16

   * - estimator
     - α
     - rel. ℓ₂
     - top-5
     - gap
     - spread
   * - mc
     - 2
     - 0.787
     - 0.84
     - 0.665
     - 0.000
   * - mc
     - 4
     - 0.562
     - 0.73
     - 0.234
     - 0.159
   * - mc
     - 8
     - 0.296
     - 0.83
     - 0.066
     - 0.079
   * - mc
     - 16
     - 0.170
     - 0.88
     - 0.027
     - 0.042
   * - mc
     - 32
     - 0.092
     - 0.93
     - 0.010
     - 0.024
   * - complementary
     - 2
     - 0.540
     - 0.76
     - 0.000
     - 0.000
   * - complementary
     - 4
     - 0.414
     - 0.70
     - 0.020
     - 0.059
   * - complementary
     - 8
     - 0.352
     - 0.80
     - 0.039
     - 0.089
   * - complementary
     - 16
     - 0.217
     - 0.85
     - 0.034
     - 0.052
   * - complementary
     - 32
     - 0.134
     - 0.90
     - 0.020
     - 0.035
   * - neyman_cc
     - 2
     - 0.540
     - 0.76
     - 0.000
     - 0.000
   * - neyman_cc
     - 4
     - 0.567
     - 0.70
     - 0.209
     - 0.095
   * - neyman_cc
     - 8
     - 0.426
     - 0.74
     - 0.065
     - 0.098
   * - neyman_cc
     - 16
     - 0.190
     - 0.86
     - 0.021
     - 0.048
   * - neyman_cc
     - 32
     - 0.109
     - 0.92
     - 0.009
     - 0.023
   * - leverage
     - 2
     - 1.249
     - 0.75
     - 0.000
     - 0.694
   * - leverage
     - 4
     - 0.537
     - 0.84
     - 0.000
     - 0.142
   * - leverage
     - 8
     - 0.270
     - 0.90
     - 0.000
     - 0.077
   * - leverage
     - 16
     - 0.150
     - 0.94
     - 0.000
     - 0.035
   * - leverage
     - 32
     - 0.081
     - 0.96
     - 0.000
     - 0.019
   * - proxy
     - 2
     - 0.746
     - 0.67
     - 0.012
     - 0.261
   * - proxy
     - 4
     - 0.465
     - 0.76
     - 0.024
     - 0.117
   * - proxy
     - 8
     - 0.222
     - 0.85
     - 0.009
     - 0.071
   * - proxy
     - 16
     - 0.107
     - 0.93
     - 0.004
     - 0.028
   * - proxy
     - 32
     - 0.054
     - 0.97
     - 0.001
     - 0.013

Banzhaf
-------

.. list-table:: Banzhaf estimand
   :header-rows: 1
   :widths: 24 8 16 16 16 16

   * - estimator
     - α
     - rel. ℓ₂
     - top-5
     - gap
     - spread
   * - mc
     - 2
     - 1.527
     - 0.84
     - 0.665
     - 0.000
   * - mc
     - 4
     - 0.980
     - 0.61
     - 0.708
     - 0.347
   * - mc
     - 8
     - 0.442
     - 0.74
     - 0.530
     - 0.121
   * - mc
     - 16
     - 0.258
     - 0.83
     - 0.471
     - 0.068
   * - mc
     - 32
     - 0.152
     - 0.89
     - 0.450
     - 0.037
   * - kernel_banzhaf
     - 2
     - 1.072
     - 0.75
     - 0.055
     - 0.713
   * - kernel_banzhaf
     - 4
     - 0.362
     - 0.86
     - 0.218
     - 0.098
   * - kernel_banzhaf
     - 8
     - 0.177
     - 0.92
     - 0.330
     - 0.047
   * - kernel_banzhaf
     - 16
     - 0.100
     - 0.95
     - 0.382
     - 0.027
   * - kernel_banzhaf
     - 32
     - 0.061
     - 0.97
     - 0.408
     - 0.017
   * - proxy
     - 2
     - 0.730
     - 0.67
     - 0.193
     - 0.192
   * - proxy
     - 4
     - 0.462
     - 0.75
     - 0.259
     - 0.123
   * - proxy
     - 8
     - 0.242
     - 0.85
     - 0.309
     - 0.068
   * - proxy
     - 16
     - 0.128
     - 0.92
     - 0.368
     - 0.033
   * - proxy
     - 32
     - 0.070
     - 0.95
     - 0.404
     - 0.018

Reading the tables
------------------

- At :math:`\alpha = 2` the paired samplers sit on their :math:`2n + 2` floor, so
  ``leverage`` is under-determined there; start at :math:`\alpha \geq 4`.
- For Shapley, ``proxy`` has the lowest error from :math:`\alpha = 8`. ``leverage``
  follows and is the only estimator whose values add up exactly at every budget;
  ``mc`` is next and needs no extra dependency. ``neyman_cc`` beats
  ``complementary`` from :math:`\alpha = 16`.
- For Banzhaf, ``kernel_banzhaf`` has the lowest error from :math:`\alpha = 4`
  (``proxy`` at the :math:`2n + 2` floor); both clearly beat ``mc``.
- The ranking depends on the game; the tables average five. Rerun the notebook on
  a game that resembles your prompts before fixing a budget.
