import TensorFlow

public protocol Estimator {
    // mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>)
}

enum EstimatorError: Error {
    case notSupportedParameter(_ msg: String)
}

public protocol Predictor {
    var scoring: String { get }

    mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>)
    func predict(data x: Tensor<Float>) -> Tensor<Float>
    func score(data x: Tensor<Float>, labels y: Tensor<Float>) -> Float
}

extension Predictor {
    public func score(data x: Tensor<Float>, labels y: Tensor<Float>) -> Float {
        guard let scorer = Scores[scoring] else {
            print("scorer not found!")
            return 0
        }

        let pred = predict(data: x)

        guard pred.shape == y.shape else {
            print("labels of predicted and expected are not in the same shape,",
                  "predicted in \(pred.shape) and expected in \(y.shape).")
            return 0
        }
        // print("y: \(y)")
        // print("pred: \(pred)")
        return scorer(y, pred)
    }
}

public protocol FitTransformer: Estimator {
    mutating func fit(_ x: Matrix)
    func transform(_ x: Matrix) -> Matrix
    func fitTranform(_ x: Matrix) -> Matrix
}
