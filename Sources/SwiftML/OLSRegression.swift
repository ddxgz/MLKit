import TensorFlow

// #if canImport(PythonKit)
//     import PythonKit
// #else
//     import Python
// #endif

protocol LinearRegressor {
    var fitIntercept: Bool { get set }
    var weights: Tensor<Float> { get set }
    var intercept_: Tensor<Float> { get }
    var coef_: Tensor<Float> { get }
    var scoring: String { get }

    mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>)

    func predict(data x: Tensor<Float>) -> Tensor<Float>

    func score(data x: Tensor<Float>, labels y: Tensor<Float>) -> Float
}

extension LinearRegressor {
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
}

func preprocessX(_ xIn: Tensor<Float>, fitIntercept: Bool) -> Tensor<Float> {
    var x: Tensor<Float> = xIn

    if fitIntercept {
        x = x.concatenated(with: Tensor<Float>(ones: [x.shape[0], 1]), alongAxis: -1)
    }
    return x
}

/// can use a dict for now, change to function with switch for more complcated cases
let Scores = ["r2": r2Score]

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
}
