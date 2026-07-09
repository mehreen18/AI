import statistics as stats

data=[2,3,4,5,6,7,5,3]

print("mean " , stats.mean(data))
print("mode " , stats.mode(data))
print("median ", stats.median(data))
print(" std // it divides by n-1 ", stats.stdev(data))
print(" pstd // it divides by n" , stats.pstdev(data))
print("multimode ", stats.multimode(data))
