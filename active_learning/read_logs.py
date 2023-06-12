## for extract results from logs
dataset = "rte"
seeds = [125, 129]
methods = ['rand-infoverse_inv']
number = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

for k in range(len(methods)):
    print("Methods:" + str(methods[k]))
    for j in range(len(seeds)):
        print("Seeds:" + str(seeds[j]))
        for i in range(len(number)):
            logs = "./models/" + str(seeds[j]) + '/' + dataset + '/' + methods[k] + '_' + str(number[i]) + "/eval_results_test_" + str(seeds[j]) + ".txt"
            f = open(logs, "r")
            line = f.readlines()
            print(line[-1])

