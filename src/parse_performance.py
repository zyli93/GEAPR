
perf = """
[Tst,2020-02-12T01:37:27.4672]  ep:6 k:10 prec_ak:0.005430 recall_ak:0.019673 mapk:0.004664
[Tst,2020-02-12T01:37:27.4672]  ep:6 k:20 prec_ak:0.005151 recall_ak:0.037155 mapk:0.005924
[Tst,2020-02-12T01:37:27.4672]  ep:6 k:30 prec_ak:0.004863 recall_ak:0.052411 mapk:0.006606
[Tst,2020-02-12T01:37:27.4672]  ep:6 k:40 prec_ak:0.004635 recall_ak:0.066730 mapk:0.007074
[Tst,2020-02-12T01:37:27.4672]  ep:6 k:50 prec_ak:0.004358 recall_ak:0.078258 mapk:0.007387
[Tst,2020-02-12T01:37:27.4672]  ep:6 k:60 prec_ak:0.004206 recall_ak:0.090635 mapk:0.007661
[Tst,2020-02-12T01:37:27.4672]  ep:6 k:70 prec_ak:0.004077 recall_ak:0.102028 mapk:0.007882
[Tst,2020-02-12T01:37:27.4672]  ep:6 k:80 prec_ak:0.003939 recall_ak:0.113048 mapk:0.008062
[Tst,2020-02-12T01:37:27.4672]  ep:6 k:90 prec_ak:0.003813 recall_ak:0.121999 mapk:0.008204
[Tst,2020-02-12T01:37:27.4672]  ep:6 k:100 prec_ak:0.003676 recall_ak:0.129991 mapk:0.008316
"""

perflist = perf.split("\n")
perflist = [x.strip().split(" ") for x in perflist]
perflist = perflist[1:-1]
print(perflist)

prl, recl, mapl = [], [], []
for x in perflist:
    print(x[4])
    prl.append(x[4].split(":")[1])
    recl.append(x[5].split(":")[1])
    mapl.append(x[6].split(":")[1])

print("Precision")
print("\t".join(prl))

print("Recall")
print("\t".join(recl))

print("MAP")
print("\t".join(mapl))



