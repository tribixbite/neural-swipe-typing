// Android Kotlin integration with ExecuTorch
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module

class MobileSwipePredictorExecutorch {
    private lateinit var module: Module
    
    fun loadModel(context: Context) {
        val modelPath = "swipe_model_android.pte"
        module = Module.load(AssetFilePath(context, modelPath))
        Log.i("SwipePredictor", "ExecuTorch model loaded successfully")
    }
    
    fun predictWord(swipePoints: List<SwipePoint>): String {
        // Extract 6D features: x, y, vx, vy, ax, ay
        val features = extractFeatures(swipePoints)
        val inputTensor = Tensor.fromBlob(
            features, 
            longArrayOf(1, features.size.toLong() / 6, 6)
        )
        
        // Run inference with ExecuTorch
        val input = EValue.from(inputTensor)
        val output = module.forward(input)
        val outputTensor = output.toTensor()
        
        // Decode fixed-length output (20 characters)
        return decodeFixedLengthOutput(outputTensor)
    }
    
    private fun extractFeatures(points: List<SwipePoint>): FloatArray {
        if (points.size < 2) return floatArrayOf()
        
        val features = mutableListOf<Float>()
        
        for (i in points.indices) {
            val p = points[i]
            
            // Normalize coordinates
            val x = p.x / 360f
            val y = p.y / 215f
            
            // Calculate velocities
            var vx = 0f
            var vy = 0f
            if (i > 0) {
                val prev = points[i - 1]
                val dt = maxOf(p.t - prev.t, 1f)
                vx = (p.x - prev.x) / dt / 1000f  // Normalized
                vy = (p.y - prev.y) / dt / 1000f
            }
            
            // Calculate accelerations
            var ax = 0f
            var ay = 0f
            if (i > 1) {
                val prev = points[i - 1]
                val prev2 = points[i - 2]
                val dt1 = maxOf(p.t - prev.t, 1f)
                val dt2 = maxOf(prev.t - prev2.t, 1f)
                val vx_prev = (prev.x - prev2.x) / dt2
                val vy_prev = (prev.y - prev2.y) / dt2
                ax = ((vx * 1000f - vx_prev) / dt1) / 500f  // Normalized
                ay = ((vy * 1000f - vy_prev) / dt1) / 500f
            }
            
            features.addAll(listOf(x, y, vx, vy, ax, ay))
        }
        
        return features.toFloatArray()
    }
    
    private fun decodeFixedLengthOutput(tensor: Tensor): String {
        val data = tensor.dataAsFloatArray
        val shape = tensor.shape()  // [1, 20, 30]
        
        var word = ""
        val seqLen = shape[1].toInt()  // 20
        val vocabSize = shape[2].toInt()  // 30
        
        for (i in 0 until seqLen) {
            var maxIdx = 0
            var maxVal = Float.NEGATIVE_INFINITY
            
            for (j in 0 until vocabSize) {
                val value = data[i * vocabSize + j]
                if (value > maxVal) {
                    maxVal = value
                    maxIdx = j
                }
            }
            
            // Convert index to character
            val char = when (maxIdx) {
                0 -> continue  // <pad>
                1 -> break     // <eos>
                2 -> '?'       // <unk>
                3 -> continue  // <sos>
                in 4..29 -> ('a' + (maxIdx - 4)).toChar()
                else -> continue
            }
            
            if (char != '?') word += char
        }
        
        return word
    }
}

data class SwipePoint(val x: Float, val y: Float, val t: Float)
