"""
compare a bunch of similar pulses

1.) Triangularity
    82127 (TRI = 0.25, low neped)  vs 82647 (TRI=0.374, high neped)
2.) Gas puff


"""

experiment_dict =
    {
        "TRIANGULARITY":
            {
                "PULSE_LOW": {'id': 82127, 'value': 0.248164},
                "PULSE_HIGH": {'id': 82647, 'value': 0.377240},
                "annotation": {'text': '$q_{95} = 3.2-3.4$\n$I_{P} = 2$ [MA]\n$B_{T} = 2$ [T]\n$P_{abs} = 10$ [MW]\nV/H divertor', 'xy': (0.68,0.70), "xycoords": 'axes fraction', 'size': 'large'},
                "profile_lims": {'PSI_LIM': (0.8, 1.05), 'T_LIM': (0, 1200)},
                "mp_dim": 4,
                "ls_dims": [1, 5, 7],
                "latex": '$\delta$'
            },
        "GASPUFF":
            {
                "PULSE_LOW": {'id': 82130, 'value': 1.12e22},
                "PULSE_HIGH": {'id': 81982, 'value': 6.95e22, "subset": [43:]},# Because there are two sets of
                "annotation": {'text': '$q_{95} = 3.2-3.4$\n$I_{P} = 2$ [MA]\n$B_{T} = 2$ [T]\n$P_{abs} = 10$ [MW]\nV/H divertor\n$\delta=0.27$', 'xy': (0.68,0.70), "xycoords": 'axes fraction', 'size': 'large'},
                "profile_lims": {'PSI_LIM': (0.8, 1.05), 'T_LIM': (-20, 1400)},
                "latex": '$\Gamma$',
                "ls_dims": [3, 6, 7],
                "mp_dim": -1,
            },
        "NBI POWER":
            {
                "PULSE_LOW": {'id': 83249, 'value': 26.5},
                "PULSE_HIGH": {'id': 83551, 'value': 17.3},
                "annotation": {'text': '$q_{95} = 3$\n$I_{P} = 2.5$ [MA]\n$B_{T} = 2$ [T]\n$\Gamma = 2-2.5$ [e/s]\nV/H divertor\n$\delta=0.28$', 'xy': (0.65,0.70), "xycoords": 'axes fraction', 'size': 'large'},
                "profile_lims": {'PSI_LIM': (0.8, 1.05), 'T_LIM': (-20, 1400)},
                "latex": '$\Gamma$',
                "ls_dims": [1, 3, 7],
                "mp_dim": -3,
            }
    }
