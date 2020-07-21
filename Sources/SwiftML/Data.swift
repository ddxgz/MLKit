import TensorFlow

public typealias Matrix = Tensor<Float>

extension Matrix {
    public func select(rows: [Int]) -> Matrix {
        // guard rows != nil || cols != nil else {
        //     print("not param provided")
        //     return self
        // }
        var new: Matrix = Matrix(repeating: 0, shape: [rows.count, self.shape[1]]) // = NdArray()
        for (i, row) in rows.enumerated() {
            // let data = x.concatenated(with: y1d, alongAxis: 1)
            // let row = self[Int(i)].reshaped(to: [1, self.shape[1]])
            // let y1d = y.reshaped(to: [y.shape[0], 1])
            // print("new: \(new.shape), row: \(row.shape)")
            // new = new.concatenated(with: row, alongAxis: 0)
            new[i, 0...] = self[row, 0...]
            // new.replacing(with: row, where: new[Int(i), 0...].Index)
            // print(row)
        }
        // return new[1..., 0...]
        return new
        // print(new[1..., 0...])
        // return data
    }

    func select(cols: [Int]) -> Matrix {
        var new: Matrix = Matrix(repeating: 0, shape: [self.shape[0], cols.count]) // = NdArray()
        // print(new.shape)
        // print(self.shape)
        for (i, col) in cols.enumerated() {
            new[0..., i] = self[0..., col]
        }
        return new
    }

    func select(rows: [Int], cols: [Int]) -> Matrix {
        var new: Matrix = Matrix(repeating: 0, shape: [rows.count, cols.count]) // = NdArray()
        // print(new.shape)
        // print(self.shape)
        for (i, row) in rows.enumerated() {
            for (j, col) in cols.enumerated() {
                new[i, j] = self[row, col]
            }
        }
        return new
    }
}
