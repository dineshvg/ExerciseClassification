import numpy as numpy

# list1 = numpy.array([1, 2, 3, 4, 5])
# list2 = numpy.array([2, 4, 6, 8, 10])
t = [[1, 2, 3, 4, 1], [1, 2, 3, 4, 3], [1, 2, 3, 4, 6], [1, 2, 3, 4, 11]]

# i = 0
# i += 1
# total = len(t)
new_t = []
for index in range(len(t)):
    # print index
    if index != 0:
        old_frame = numpy.array(t[index-1])
        current_frame = numpy.array(t[index])
        gradient_change = (current_frame - old_frame)**2
        print gradient_change
        max_grad = max(gradient_change)
        gradient_change = [x / max_grad for x in gradient_change]
        new_t.append(gradient_change)

print new_t
    # if index == 0:
    #     new_t.append(t[index])
    # if index == len(t)-1:
    #     new_t.append(t[index])




# temp_list = numpy.array(list)
# some_list = [list1, list2, "barry"]
#
# for item in some_list:
#     print "Item - " + item
#     if some_list.index(item) != len(some_list) -1:
#         next_item = some_list[some_list.index(item) + 1]
#         print "Next item -", next_item
#     else:
#         print "Next item does not exist!"