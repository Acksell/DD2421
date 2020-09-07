import monkdata as m
import dtree
from drawtree_qt5 import drawTree 

def calculate_entropies():
    print(dtree.entropy(m.monk1))
    print(dtree.entropy(m.monk2))
    print(dtree.entropy(m.monk3))
    

def calculate_average_gains():
    datasets = [m.monk1, m.monk2, m.monk3]
    table = []
    for j in range(len(datasets)):
        row = [] 
        for i, attribute in enumerate(m.attributes):
            row.append(dtree.averageGain(datasets[j], attribute))
        table.append(row)
    return table

# for row in calculate_average_gains():
#   print(row)

tree1 = dtree.buildTree(m.monk1, m.attributes)
tree2 = dtree.buildTree(m.monk2, m.attributes)
tree3 = dtree.buildTree(m.monk3, m.attributes)

print("Training accuracy monk1", dtree.check(tree1, m.monk1))
print("Training accuracy monk2", dtree.check(tree2, m.monk2))
print("Training accuracy monk3", dtree.check(tree3, m.monk3))

print("Test set accuracy monk1", dtree.check(tree1, m.monk1test))
print("Test set accuracy monk2", dtree.check(tree2, m.monk2test))
print("Test set accuracy monk3", dtree.check(tree3, m.monk3test))

# drawTree(tree)
