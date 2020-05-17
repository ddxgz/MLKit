import TensorFlow

/// can use a dict for now, change to function with switch for more complcated
/// cases
let Scores = ["r2": r2Score,
              "accuracy": accuracy]

func accuracy(_ y: Tensor<Float>, _ pred: Tensor<Float>) -> Float {
    // let y1d = y.reshaped(to: [y.shape[0], 1])
    // print(y.shape)
    // print(y1d.shape)
    // print(pred.shape)
    var cnt = 0.0
    for i in 0 ..< y.shape[0] {
        if y[i] == pred[i] {
            cnt += 1
        }
    }
    return Float(cnt / Double(y.shape[0]))
}

func r2Score(_ y: Tensor<Float>, _ pred: Tensor<Float>) -> Float {
    // func r2Score(true y: Tensor<Float>, pred predicted: Tensor<Float>) -> Float {
    let SS_res = pow(y - pred, 2).sum()
    print(SS_res)
    let SS_tot = pow(y - y.mean(), 2).sum()
    print(SS_tot)
    print(SS_res / SS_tot)
    let score = 1 - (SS_res / SS_tot)
    return Float(score.scalarized())
}
