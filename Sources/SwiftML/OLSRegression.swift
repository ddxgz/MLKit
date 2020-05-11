import Foundation
import TensorFlow

// import PythonKit
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

protocol LinearRegressor {
    var fitIntercept: Bool { get set }
    var weights: Tensor<Float> { get set }
    var intercept_: Tensor<Float> { get }
    var coef_: Tensor<Float> { get }

    mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>)

    func predict(data x: Tensor<Float>) -> Tensor<Float>

    func score(data x: Tensor<Float>, labels y: Tensor<Float>) -> Float
}

func preprocessX(_ xIn: Tensor<Float>, fitIntercept: Bool) -> Tensor<Float> {
    var x: Tensor<Float> = xIn

    if fitIntercept {
        x = x.concatenated(with: Tensor<Float>(ones: [x.shape[0], 1]), alongAxis: -1)
    }
    return x
}

func r2Score(_ y: Tensor<Float>, _ predicted: Tensor<Float>) -> Float {
    // func r2Score(true y: Tensor<Float>, pred predicted: Tensor<Float>) -> Float {
    let SS_res = pow(y - predicted, 2).sum()
    print(SS_res)
    let SS_tot = pow(y - y.mean(), 2).sum()
    print(SS_tot)
    print(SS_res / SS_tot)
    let score = 1 - (SS_res / SS_tot)
    return Float(score.scalarized())
}

/// can use a dict for now, change to function with switch for more complcated cases
let Scores = ["r2": r2Score]
// enum Scores {
//     case r2
// }

/// Linear regression implements Ordinary Least Squares.
///
/// Can be called by `model.fit(x, y)` or `model(x, y)`.
struct OLSRegression: LinearRegressor {
    var fitIntercept: Bool
    var scoring: String
    var weights: Tensor<Float>
    var intercept_: Tensor<Float> {
        if fitIntercept {
            return weights[-1, 0 ... weights.shape[1]]
        } else {
            return Tensor(0)
        }
    }

    var coef_: Tensor<Float> {
        if fitIntercept {
            return weights[
                0 ..< weights.shape[0] - 1, 0 ... weights.shape[1]
            ]
        } else {
            return weights
        }
    }

    init(fitIntercept: Bool = true, scoring: String = "r2") {
        weights = Tensor<Float>(0)
        self.fitIntercept = fitIntercept
        // precondition(Scores.keys.contains(scoring),
        //              "scoring \(scoring) not supported!")
        self.scoring = scoring
    }

    mutating func callAsFunction(data x: Tensor<Float>, labels y: Tensor<Float>) {
        fit(data: x, labels: y)
    }

    // \beta = (X^T dot X)^-1 dot X^T dot y
    mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>) {
        // var x = x

        // if self.fitIntercept {
        //     x = x.concatenated(with: Tensor<Float>(ones: [x.shape[0], 1]),
        //                         alongAxis: -1)
        // }
        let x = preprocessX(x, fitIntercept: fitIntercept)

        weights = matmul(
            matmul(
                _Raw.matrixInverse(
                    matmul(x.transposed(), x)),
                x.transposed()
            ),
            y
        )
    }

    func predict(data x: Tensor<Float>) -> Tensor<Float> {
        // let x = preprocessX(x)
        let x = preprocessX(x, fitIntercept: fitIntercept)
        return matmul(x, weights)
    }

    // r^2
    func score(data x: Tensor<Float>, labels y: Tensor<Float>) -> Float {
        let predicted = predict(data: x)
        guard let scorer = Scores[scoring] else {
            print("scorer not found")
            return 0
        }
        return scorer(y, predicted)
    }

    // func preprocessX(_ x: Tensor<Float>) -> Tensor<Float> {
    //     var x = x

    //     if self.fitIntercept {
    //         x = x.concatenated(with: Tensor<Float>(ones: [x.shape[0], 1]), alongAxis: -1)
    //     }
    //     return x
    // }
}

let np = Python.import("numpy")
let datasets = Python.import("sklearn.datasets")

let diabetes = datasets.load_diabetes()

let diabetesData = Tensor<Float>(numpy: np.array(diabetes.data, dtype: np.float32))!
var diabetesLabels = Tensor<Float>(numpy: np.array(diabetes.target, dtype: np.float32))!

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
