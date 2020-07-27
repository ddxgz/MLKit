import XCTest

import MLKit
// import LASwift
import Nimble
import TensorFlow

// import PythonKit
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

let np = Python.import("numpy")
let datasets = Python.import("sklearn.datasets")
let sktree = Python.import("sklearn.tree")

final class SwiftMLTests: XCTestCase {
    override func setUp() {
        try! FileManager().createDirectory(atPath: "Tests/tmp", withIntermediateDirectories: true)
    }

    override func tearDown() {
        try! FileManager().removeItem(atPath: "Tests/tmp")
    }

    static var allTests = [
        ("testPCA", testPCA),
        // ("testMatrix", testMatrix),
        ("testOLS", testOLS),
    ]

    func testPCA() throws {
        let cnt = 5
        // let m1 = Tensor(randomNormal: [cnt, cnt], mean: Tensor<Float>(5),
        // standardDeviation: Tensor<Float>(5))
        let m1 = Tensor(shape: [3, 5], scalars: [Float](stride(from: 0.0, to: 15.0, by: 1.0)))

        var pca = try PCA(nComponents: 3, svdSolver: "full")
        pca.fit(m1)
        print(pca.components)
        print(pow(pca.components!, 2))
        print(pca.explainedVarianceRatio)
        print(pca.singularValues)

        let decom = Python.import("sklearn.decomposition")
        var pcaSk = decom.PCA(n_components: 3, svd_solver: "full")
        pcaSk.fit(m1.makeNumpyArray())
        let componentsSk = Tensor<Float>(numpy: pcaSk.components_)
        print(componentsSk)

        // // let (u, s, v) = m1.svd()
        // let (u, s, v) = _Raw.svd(m1, fullMatrices: false)
        // print("tf shape: \(u.shape), \(s.shape), \(v.shape)")
        // print(Matrix(v.transposed())[0..., 0 ..< 3])

        // let linalg = Python.import("scipy.linalg")
        // let svdobj = linalg.svd(m1.makeNumpyArray(), full_matrices: false)
        // print("scipy shape: \(svdobj[0].shape), \(svdobj[1].shape), \(svdobj[2].shape)")
        // print(svdobj[2][0 ..< 3])

        // print(m1.svd(fullMatrices: false).v)

        // let vSci = Tensor<Float>(numpy: svdobj[2])
        // expect(Matrix(v)) == vSci

        expect(pca.components) == componentsSk
        // print("PCA components are equal")

        let m2 = Tensor(shape: [3, 5], scalars: [Float](stride(from: 2.0, to: 17.0, by: 1.0)))
        print(try! pca.transform(m2))

        var pca2 = try PCA(nComponents: 3, svdSolver: "full")
        print(pca2.fitTranform(m2))
    }

    // func testMatrix() throws {
    //     let cnt = 5
    //     let m1 = ones(cnt, cnt)
    //     print(m1)
    //     XCTAssertEqual(m1.rows, cnt)

    //     let (u, s, v) = svd(m1)
    //     print(u, s, v)
    // }

    func testOLS() throws {
        let np = Python.import("numpy")
        // let datasets = Python.import("sklearn.datasets")
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

    func testGDRegression() {
        let np = Python.import("numpy")
        // let datasets = Python.import("sklearn.datasets")
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
        print("score: \(score)")

        var gdmodel = GDRegression(
            learningRate: 0.5, epochs: 2000, optimizer: "sgd"
        )
        gdmodel.fit(data: trainData, labels: trainLabels)
        let gdscore = gdmodel.score(data: testData, labels: testLabels)
        print("gdscore: \(gdscore)")
        print("weights: \(gdmodel.weights)")
    }

    func test_tree_classification() {
        // let diabetes = datasets.load_diabetes()
        // let diabetes = datasets.load_iris()
        let diabetes = datasets.load_breast_cancer()

        // let diabetesData = Tensor<Float>(numpy: np.array(diabetes.data, dtype: np.float32))!
        // let diabetesLabels = Tensor<Float>(numpy: np.array(diabetes.target, dtype: np.float32))!

        // // let data = diabetesData.slice(lowerBounds: [0, 0],
        // //                               upperBounds: [diabetesData.shape[0], 3])
        // let data = diabetesData
        // let labels = diabetesLabels.reshaped(to: [diabetesLabels.shape[0], 1])

        let diabetesData = Matrix(numpy: np.array(diabetes.data, dtype: np.float32))!
        let diabetesLabels = Matrix(numpy: np.array(diabetes.target, dtype: np.float32))!

        let shuffledIdx = [Int](np.random.permutation(diabetesData.shape[0]))!
        // print("shuffled: \(shuffledIdx)")
        // print(diabetesData)
        let data = diabetesData.select(rows: shuffledIdx)
        let labels = diabetesLabels.reshaped(to: [diabetesLabels.shape[0], 1]).select(rows: shuffledIdx)

        let start = 0
        // let dataLen = data.shape[0]
        let dataLen = 150
        let cols = 10
        let test_size = 0.3
        let testLen = Int(Double(dataLen) * test_size)
        let trainEnd = start + dataLen - testLen

        let trainData = data.slice(lowerBounds: [start, 0], upperBounds: [trainEnd, cols])
        let testData = data.slice(lowerBounds: [trainEnd, 0],
                                  upperBounds: [trainEnd + testLen, cols])

        let trainLabels = labels.slice(lowerBounds: [start, 0], upperBounds: [trainEnd, 1])
        let testLabels = labels.slice(lowerBounds: [trainEnd, 0],
                                      upperBounds: [trainEnd + testLen, 1])

        // print(trainData)
        // print(trainLabels)
        // let trainDataset: Dataset<IrisBatch> = Dataset(
        //     contentsOfCSVFile: trainDataFilename, hasHeader: true,
        //     featureColumns: [0, 1, 2, 3], labelColumns: [4]
        // ).batched(batchSize)

        var model = DecisionTreeClassifier()
        // var model = OLSRegression(fitIntercept: true)
        model.fit(data: trainData, labels: trainLabels)
        model.printTree()
        // model.predict(data: testData)
        print(testData)
        print(testLabels)
        let score = model.score(data: testData, labels: testLabels)
        print("score: \(score)")

        // let skmodel = sktree.DecisionTreeRegressor()
        let skmodel = sktree.DecisionTreeClassifier()
        skmodel.fit(trainData.makeNumpyArray(), trainLabels.makeNumpyArray())
        let scoreSK = skmodel.score(testData.makeNumpyArray(), testLabels.makeNumpyArray())
        print("skscore: \(scoreSK)")
    }

    func test_tree_regression() {
        // let diabetes = datasets.load_diabetes()
        let diabetes = datasets.fetch_california_housing()
        // let diabetes = datasets.load_boston()
        // let diabetes = datasets.load_iris()

        let diabetesData = Matrix(numpy: np.array(diabetes.data, dtype: np.float32))!
        let diabetesLabels = Matrix(numpy: np.array(diabetes.target, dtype: np.float32))!

        let shuffledIdx = [Int](np.random.permutation(diabetesData.shape[0]))!
        // print("shuffled: \(shuffledIdx)")
        // print(diabetesData)
        let data = diabetesData.select(rows: shuffledIdx)
        let labels = diabetesLabels.reshaped(to: [diabetesLabels.shape[0], 1]).select(rows: shuffledIdx)
        // print(data)
        let start = 0
        // let dataLen = data.shape[0]
        let dataLen = 60
        let cols = 5
        let test_size = 0.3
        let testLen = Int(Double(dataLen) * test_size)
        let trainEnd = start + dataLen - testLen

        let trainData = data.slice(lowerBounds: [start, 0], upperBounds: [trainEnd, cols])
        let testData = data.slice(lowerBounds: [trainEnd, 0],
                                  upperBounds: [trainEnd + testLen, cols])

        let trainLabels = labels.slice(lowerBounds: [start, 0], upperBounds: [trainEnd, 1])
        let testLabels = labels.slice(lowerBounds: [trainEnd, 0],
                                      upperBounds: [trainEnd + testLen, 1])

        print(trainData)
        print(trainLabels)
        // let trainDataset: Dataset<IrisBatch> = Dataset(
        //     contentsOfCSVFile: trainDataFilename, hasHeader: true,
        //     featureColumns: [0, 1, 2, 3], labelColumns: [4]
        // ).batched(batchSize)

        print(testData)
        print(testLabels)

        var model = DecisionTreeRegressor(criterion: "mse", minSamplesSplit: 2, minSamplesLeaf: 1, scoring: "r2")
        model.fit(data: trainData, labels: trainLabels)
        model.printTree()
        // model.predict(data: testData)
        let score = model.score(data: testData, labels: testLabels)
        print("score: \(score)")

        var modelols = OLSRegression(fitIntercept: true)
        modelols.fit(data: trainData, labels: trainLabels)
        // let predOLS = model.predict(data: testData)
        let scoreols = modelols.score(data: testData, labels: testLabels)
        print("scoreols: \(scoreols)")
        // print("predOLS: \(predOLS)")

        let skmodel = sktree.DecisionTreeRegressor()
        skmodel.fit(trainData.makeNumpyArray(), trainLabels.makeNumpyArray())
        // print(testData.shape, testLabels.shape)
        let predSk = skmodel.predict(testData.makeNumpyArray())
        let scoreSK = skmodel.score(testData.makeNumpyArray(), testLabels.makeNumpyArray())
        print("skscore: \(scoreSK)")
        print("sk pred: \(predSk)")
        let path = skmodel.decision_path(testData.makeNumpyArray())
        // print("path: \(path)")
        print(skmodel.feature_importances_)
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

        var model = DecisionTreeClassifier()
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
}
