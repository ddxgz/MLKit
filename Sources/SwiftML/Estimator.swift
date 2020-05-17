import TensorFlow

protocol Estimator {
    // var featureImportances: Tensor<Float> { get }
    var scoring: String { get }

    mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>)

    func predict(data x: Tensor<Float>) -> Tensor<Float>

    func score(data x: Tensor<Float>, labels y: Tensor<Float>) -> Float
}

extension Estimator {
    func score(data x: Tensor<Float>, labels y: Tensor<Float>) -> Float {
        let pred = predict(data: x)

        guard let scorer = Scores[scoring] else {
            print("scorer not found")
            return 0
        }
        return scorer(y, pred)
    }
}
