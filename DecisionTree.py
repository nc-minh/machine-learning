from sklearn import tree


# step 1:   Thu thap du lieu 
# step 2:   Xu ly du lieu
# step 3:   Xay dung model
# step 4:   Du doan ket qua
# step 5:   Danh gia

my_stree = tree.DecisionTreeClassifier()

m_featured = [
    [1, 2, 2, 3],
    [3, 1, 3, 1],
    [1, 1, 3, 1],
    [3, 3, 3, 2],
    [1, 1, 1, 3],
    [2, 1, 2, 3],
    [2, 2, 2, 1],
    [3, 1, 1, 3]
]

m_label = [0, 1, 1, 0 ,0 ,0 , 0, 1]

result = my_stree.fit(m_featured, m_label)

end_result = result.predict([[3, 1, 2, 3]])

print(end_result)