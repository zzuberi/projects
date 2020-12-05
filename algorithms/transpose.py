test = [[x for x in range(5)], [x for x in range(5,10)]]
new = []

print(test)
# for i in range(len(test[0])):
#     for j in range(i, len(test)):
#         test[i][j], test[j][i] = test[j][i], test[i][j]


print(list(map(list,zip(*test))))
test = list(map(list,zip(*test[::-1])))
print(test)
