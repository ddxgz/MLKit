import Foundation
import TensorFlow

// import PythonKit
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

// import SwiftML

// typealias IntValue = Int32

// import SwiftML
let np = Python.import("numpy")
let datasets = Python.import("sklearn.datasets")
let sktree = Python.import("sklearn.tree")

func testOLS() {
    let np = Python.import("numpy")
    let datasets = Python.import("sklearn.datasets")
    let diabetes = datasets.load_diabetes()

    let diabetesData = Tensor<Float>(numpy: np.array(diabetes.data, dtype: np.float32))!
    let diabetesLabels = Tensor<Float>(numpy: np.array(diabetes.target, dtype: np.float32))!

    let data = diabetesData.slice(lowerBounds: [0, 0], upperBounds: [diabetesData.shape[0], diabetesData.shape[1]])
    let labels = diabetesLabels.reshaped(to: [diabetesLabels.shape[0], 1])

    let dataLen = data.shape[0]
    let dataWid = data.shape[1]
    let test_size = 0.3
    let testLen = Int(Double(data.shape[0]) * test_size)
    let trainEnd = dataLen - testLen

    let trainData = data.slice(lowerBounds: [0, 0], upperBounds: [trainEnd, dataWid])
    let testData = data.slice(lowerBounds: [trainEnd, 0], upperBounds: [trainEnd + testLen, dataWid])

    let trainLabels = labels.slice(lowerBounds: [0, 0], upperBounds: [trainEnd, 1])
    let testLabels = labels.slice(lowerBounds: [trainEnd, 0], upperBounds: [trainEnd + testLen, 1])

    var model = OLSRegression(fitIntercept: true)
    // model.fit(data: trainData, labels: trainLabels)
    model(data: trainData, labels: trainLabels)
    // model.predict(data: testData)
    print(model.weights)
    // print(model.weights.shape)
    print(model.coef_)
    print(model.intercept_)
    let score = model.score(data: testData, labels: testLabels)
    print(score)
}

func test_tree_regression_2() {
    // let diabetes = datasets.load_diabetes()
    let diabetes = datasets.load_iris()

    let diabetesData = Tensor<Float>(numpy: np.array(diabetes.data, dtype: np.float32))!
    let diabetesLabels = Tensor<Float>(numpy: np.array(diabetes.target, dtype: np.float32))!

    let data = diabetesData.slice(lowerBounds: [0, 0],
                                  upperBounds: [diabetesData.shape[0], 3])
    let labels = diabetesLabels.reshaped(to: [diabetesLabels.shape[0], 1])

    let start = 16
    // let dataLen = data.shape[0]
    let dataLen = 90
    let test_size = 0.3
    let testLen = Int(Double(dataLen) * test_size)
    let trainEnd = start + dataLen - testLen

    let trainData = data.slice(lowerBounds: [start, 0], upperBounds: [trainEnd, 3])
    // let testData = data.slice(lowerBounds: [trainEnd, 0],
    //                           upperBounds: [trainEnd + testLen, 3])

    let trainLabels = labels.slice(lowerBounds: [start, 0], upperBounds: [trainEnd, 1])
    // let testLabels = labels.slice(lowerBounds: [trainEnd, 0],
    //                               upperBounds: [trainEnd + testLen, 1])

    // print(trainData)
    // print(trainLabels)
    // let trainDataset: Dataset<IrisBatch> = Dataset(
    //     contentsOfCSVFile: trainDataFilename, hasHeader: true,
    //     featureColumns: [0, 1, 2, 3], labelColumns: [4]
    // ).batched(batchSize)

    var model = DecisionTree()
    // var model = OLSRegression(fitIntercept: true)
    model.fit(data: trainData, labels: trainLabels)
    model.printTree()
    // model.predict(data: testData)
    // // print(model.weights)
    // // // print(model.weights.shape)
    // // print(model.coef_)
    // // print(model.intercept_)
    // // let score = model.score(data: testData, labels: testLabels)
    // // print(score)
    // var skmodel = sktree.DecisionTreeClassifier()
    // skmodel.fit(trainData.makeNumpyArray(), trainLabels.makeNumpyArray())
    // print(skmodel.tree_.decision_path(trainData.makeNumpyArray()))
}

func test_gini_2() {
    // let dataset = np.array([[2.771244718, 1.784783929, 0],
    //                         [1.728571309, 1.169761413, 0],
    //                         [3.678319846, 2.81281357, 0],
    //                         //  [8.961043357, 2.61995032, 0],
    //                         [3.961043357, 2.61995032, 0],
    //                         [2.999208922, 2.209014212, 0],
    //                         [7.497545867, 3.162953546, 1],
    //                         [9.00220326, 3.339047188, 1],
    //                         [7.444542326, 0.476683375, 1],
    //                         [10.12493903, 3.234550982, 1],
    //                         [6.642287351, 3.319983761, 1]])

    // let col = 2
    // print(dataset[0, col])
    let dataset = Tensor<Float>([[2.771244718, 1.784783929, 0],
                                 [1.728571309, 1.169761413, 0],
                                 
                                 [2.999208922, 2.209014212, 1],
                                 
                                 [3.678319846, 2.81281357, 0],
                                 
                                 [3.961043357, 2.61995032, 1],
                                 [6.642287351, 3.319983761, 1],
                                 [7.444542326, 0.476683375, 1],
                                 [7.497545867, 3.162953546, 1],
                                 
                                 [8.961043357, 2.61995032, 0],
                                 
                                 [9.00220326, 3.339047188, 1],
                                 [10.12493903, 3.234550982, 1]])

    let features = dataset[0..., 0 ... 1]
    let labels = dataset[0..., 2]
    print(features)
    print(labels)

    var model = DecisionTree()
    // var model = OLSRegression(fitIntercept: true)
    model.fit(data: features, labels: labels)
    // model.printTree()

    let testdata = Tensor<Float>([[1.771244718, 1.784783929],
                                  [1.928571309, 1.169761413],
                                  
                                  [3.861043357, 2.61995032],
                                  [6.942287351, 3.319983761],
                                  [8.444542326, 0.476683375],
                                  [11.12493903, 3.234550982]])
    let pred = model.predict(data: testdata)
    print("pred: \(pred)")
    let score = model.score(data: testdata, labels: [0, 0, 1, 0, 1, 1])
    print("score: \(score)")
    model.printTree()

    // let mat = Matrix(dataset)
    // print(mat.select(rows: [0, 2, 4]))
}

// test_tree_regression()
test_gini_2()
