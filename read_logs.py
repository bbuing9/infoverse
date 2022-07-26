## for extract results from logs

seeds = [1234, 2345, 3456, 4567]
methods = ['info_dens_dpp']
ratio = [0.83, 0.66, 0.5, 0.33, 0.25, 0.17, 0.13, 0.09, 0.05]

for k in range(len(methods)):
    print("Method:" + str(methods[k]))
    for j in range(len(ratio)):
        print("Ratio:" + str(ratio[j]))
        for i in range(len(seeds)):
            logs = "./logs/rte_R1.0_roberta_large_1231_base_" + methods[k] + '_N' + str(ratio[j]) + '_S' + str(seeds[i]) + "/log.txt"
            f = open(logs, "r")
            line = f.readlines()
            print(line[-1])

