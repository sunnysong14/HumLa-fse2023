from data_stream.real_data_stream import data_id_2name
from main import run_HumLa, compute_human_PFs, compute_human_churn
from datetime import datetime


VERBOSE, IS_PLOT = 0, False  # for run_HumLa
print("Please note that 100 runs with 10k test steps can take lots of time")
print("So one can set smaller N_TEST and SEEDs to have a quick check")
N_TEST, SEEDS = 10000, range(100)  # setup in the manuscript
# N_TEST, SEEDS = 1000, range(2)  # for a quick run


def RQ1(str_RQ1="RQ1.1", project_id=0):
    """ HumLa
    Replication of RQ1 that runs the waiting-time method and HumLa at varying levels of human noise and human effort.
    Input arguments:
        str_RQ1:        "RQ1.1", "RQ1.2", and "RQ1.3"
        project_id:     0, 1, ..., 13
    """

    str_RQ1 = str_RQ1.upper()
    if str_RQ1 == "RQ1.1":  # RQ1.1: HumLa vs waiting-time method
        print("\n" + "==" * 30)
        print("RQ1.1 for %s: PF of JIT-SDP with the waiting-time method vs HumLa:"
              % data_id_2name(project_id))
        # waiting-time method
        run_HumLa({"has_human": False}, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT)
        # HumLa
        run_HumLa({"has_human": True, "human_err": 0, "human_eff": 1}, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT)

    elif str_RQ1 == "RQ1.2":  # RQ1.2: HumLa at varying levels of human noise
        print("\n" + "==" * 30)
        print("RQ1.2 for %s: average PF of HumLa at different human noise across test steps:"
              % data_id_2name(project_id))
        human_noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for hm, noise in enumerate(human_noises):
            human_dict = {"has_human": True, "human_err": noise, "human_eff": 1}
            run_HumLa(human_dict, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT)

    elif str_RQ1 == "RQ1.3":  # RQ1.3: HumLa at varying levels of human effort
        print("\n" + "==" * 30)
        print("RQ1.3 for %s: PF of HumLa at different human efforts:"
              % data_id_2name(project_id))
        human_efforts = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for hm, effort in enumerate(human_efforts):
            human_dict = {"has_human": True, "human_err": 0, "human_eff": effort}
            run_HumLa(human_dict, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT)
    else:
        raise Exception("Error in str_RQ=%s." % str_RQ1)


def RQ2(str_RQ2="RQ2.1", project_id=0):
    """ Eco-HumLa
    Replicate RQ2 that investigates the proposed Eco-HumLa.

    Input argument:
        str_RQ2:        "RQ2.1", "RQ2.2", and "RQ2.3"
        project_id:     0, 1, ..., 13
    """
    str_RQ2 = str_RQ2.upper()
    if str_RQ2 == "RQ2.1":  # RQ2.1: Eco-HumLa
        print("\n" + "==" * 30)
        print("RQ2.1 for %s: PF of Eco-HumLa vs HumLa at different efforts across #seeds=%d"
              % (data_id_2name(project_id), len(SEEDS)))
        # Eco-HumLa
        human_dict = {"has_human": True, "human_err": 0, "human_eff": "auto_ecohumla2"}
        run_HumLa(human_dict, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT)

        # HumLa at different human efforts
        human_efforts = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for hm, effort in enumerate(human_efforts):
            human_dict = {"has_human": True, "human_err": 0, "human_eff": effort}
            run_HumLa(human_dict, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT)

    elif str_RQ2 == "RQ2.2":  # RQ2.2: cumulative code churns from HumLa and Eco-HumLa
        print("\n" + "==" * 30)
        print("RQ2.2 for %s: cumulative code churn (in kilo) across #seeds=%d"
              % (data_id_2name(project_id), len(SEEDS)))
        # Eco-HumLa
        compute_human_churn("auto_ecohumla2", project_id, N_TEST, SEEDS)

        # HumLa at different levels of human effort
        human_efforts = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for hm, effort in enumerate(human_efforts):
            compute_human_churn(effort, project_id, N_TEST, SEEDS)

    elif str_RQ2 == "RQ2.3":  # RQ2.3: recall 1 and false alarm for humans to investigate
        # ECo-HumLa
        eco_humla_recall1, eco_humla_false_alarm = compute_human_PFs("auto_ecohumla2", project_id, N_TEST, SEEDS)
        # HumLa at 50%-human effort
        humla50_recall1, humla50_false_alarm = compute_human_PFs(0.5, project_id, N_TEST, SEEDS)

        print("\n" + "==" * 25)
        print("RQ2.3 for %s: recall-1 and false-alarm for humans across #seeds=%d"
              % (data_id_2name(project_id), len(SEEDS)))
        print("\t Eco-HumLa: recall-1 = %.2f, false-alarm = %.2f" % (eco_humla_recall1, eco_humla_false_alarm))
        print("\t HumLa at 50%%-human effort: recall 1 = %.2f, false alarm = %.2f" % (humla50_recall1, humla50_false_alarm))

    else:
        raise Exception("Error in str_RQ=%s." % str_RQ2)


if __name__ == "__main__":
    project_id = 0
    RQ1("RQ1.1", project_id)
    RQ1("RQ1.2", project_id)
    RQ1("RQ1.3", project_id)
    RQ2("RQ2.1", project_id)
    RQ2("RQ2.2", project_id)
    RQ2("RQ2.3", project_id)
    print("\nSuccess on %s\n" % datetime.today())

