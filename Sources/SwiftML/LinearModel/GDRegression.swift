import TensorFlow

// TODO: add config for activation, optimizer, loss
struct GDRegression: LinearRegressor {
    var fitIntercept: Bool = true
    var scoring: String = "r2"
    var weights: Tensor<Float>
    var intercept_: Tensor<Float> { self.model.bias }
    var coef_: Tensor<Float> { self.model.weight }

    var learningRate: Float
    var epochs: Int
    var model: Dense<Float>

    init(nFeatures: Int, learningRate: Float, epochs: Int) {
        self.learningRate = learningRate
        self.epochs = epochs

        self.weights = Tensor<Float>(0)
        // self.scoring = "r2"
        self.model = Dense<Float>(inputSize: nFeatures, outputSize: 1,
                                  activation: relu)
    }

    // @differentiable
    mutating func fit(data x: Tensor<Float>, labels y: Tensor<Float>) {
        let optim = SGD(for: self.model, learningRate: self.learningRate)
        // let optim = RMSProp(for: self.model, learningRate: self.learningRate)
        Context.local.learningPhase = .training

        for i in 0 ..< self.epochs {
            let modelGrad = gradient(at: self.model) { model -> Tensor<Float> in
                let yhat = model(x)
                let loss = l1Loss(predicted: yhat, expected: y)
                // let loss = meanSquaredError(predicted: yhat, expected: y)
                // print("loss: \(loss)")
                return loss
            }
            optim.update(&self.model, along: modelGrad)
        }
        self.weights = self.model.weight.concatenated(
            with: self.model.bias.reshaped(to: [1, 1]),
            alongAxis: 0
        )
    }

    func predict(data x: Tensor<Float>) -> Tensor<Float> {
        return self.model(x)
    }
}
