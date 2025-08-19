package com.example.realtime_object

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.realtime_object.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var interpreter: Interpreter
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val inputSize = 640
    private var previewWidth = 0
    private var previewHeight = 0

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val IOU_THRESHOLD = 0.45f
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initializeModel()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun initializeModel() {
        try {
            val modelBuffer = FileUtil.loadMappedFile(this, "best_float32.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseNNAPI(false) // Set to true if you want to use NNAPI
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.d("MainActivity", "Model loaded successfully")
        } catch (e: Exception) {
            Log.e("MainActivity", "Error loading TFLite model", e)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                            .build()
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor, ImageAnalyzer())
                    }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

            } catch (e: Exception) {
                Log.e("MainActivity", "Error starting camera", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        @OptIn(ExperimentalGetImage::class)
        override fun analyze(imageProxy: ImageProxy) {
            try {
                val bitmap = imageProxyToBitmap(imageProxy)
                if (bitmap != null) {
                    // Store preview dimensions for coordinate scaling
                    previewWidth = imageProxy.width
                    previewHeight = imageProxy.height

                    val detections = detectObjects(bitmap)
                    val scaledDetections = scaleDetections(detections, bitmap.width, bitmap.height)

                    runOnUiThread {
                        binding.overlayView.setResults(
                            scaledDetections.map { it.rect },
                            scaledDetections.map { "Printer" },
                            scaledDetections.map { it.confidence }
                        )
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error in image analysis", e)
            } finally {
                imageProxy.close()
            }
        }
    }

    @OptIn(ExperimentalGetImage::class) private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val image = imageProxy.image ?: return null

            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
            val imageBytes = out.toByteArray()
            var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

            // Handle rotation
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            if (rotationDegrees != 0) {
                val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            }

            // Resize to model input size
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        } catch (e: Exception) {
            Log.e("MainActivity", "Error converting image", e)
            null
        }
    }

    private fun detectObjects(bitmap: Bitmap): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            val inputBuffer = convertBitmapToByteBuffer(bitmap)

            // YOLO output format: [batch, features, boxes] = [1, 5, 8400]
            // Changed from [1, 8400, 5] to [1, 5, 8400] to match model output
            val output = Array(1) { Array(5) { FloatArray(8400) } }

            interpreter.run(inputBuffer, output)

            // Parse YOLO output - now accessing [batch][feature][box] instead of [batch][box][feature]
            for (i in 0 until 8400) {
                val x = output[0][0][i] // center x (normalized)
                val y = output[0][1][i] // center y (normalized)
                val w = output[0][2][i] // width (normalized)
                val h = output[0][3][i] // height (normalized)
                val confidence = output[0][4][i] // confidence score

                if (confidence > CONFIDENCE_THRESHOLD) {
                    // Convert from center coordinates to corner coordinates
                    val xMin = (x - w / 2f) * inputSize
                    val yMin = (y - h / 2f) * inputSize
                    val xMax = (x + w / 2f) * inputSize
                    val yMax = (y + h / 2f) * inputSize

                    // Clamp to image boundaries
                    val left = max(0f, xMin)
                    val top = max(0f, yMin)
                    val right = min(inputSize.toFloat(), xMax)
                    val bottom = min(inputSize.toFloat(), yMax)

                    if (right > left && bottom > top) {
                        detections.add(Detection(RectF(left, top, right, bottom), confidence))
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error in object detection", e)
        }

        // Apply Non-Maximum Suppression
        return applyNMS(detections, IOU_THRESHOLD)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        val intValues = IntArray(inputSize * inputSize)
        bitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in intValues) {
            // Normalize pixel values to [0,1]
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
        return buffer
    }

    private fun scaleDetections(detections: List<Detection>, modelWidth: Int, modelHeight: Int): List<Detection> {
        val overlayWidth = binding.overlayView.width.toFloat()
        val overlayHeight = binding.overlayView.height.toFloat()

        if (overlayWidth == 0f || overlayHeight == 0f) return detections

        val scaleX = overlayWidth / modelWidth
        val scaleY = overlayHeight / modelHeight

        return detections.map { detection ->
            val scaledRect = RectF(
                detection.rect.left * scaleX,
                detection.rect.top * scaleY,
                detection.rect.right * scaleX,
                detection.rect.bottom * scaleY
            )
            Detection(scaledRect, detection.confidence)
        }
    }

    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<Detection>()

        for (detection in sortedDetections) {
            var shouldAdd = true
            for (selectedDetection in selectedDetections) {
                if (calculateIoU(detection.rect, selectedDetection.rect) > iouThreshold) {
                    shouldAdd = false
                    break
                }
            }
            if (shouldAdd) {
                selectedDetections.add(detection)
            }
        }

        return selectedDetections
    }

    private fun calculateIoU(rect1: RectF, rect2: RectF): Float {
        val intersectionArea = max(0f, min(rect1.right, rect2.right) - max(rect1.left, rect2.left)) *
                max(0f, min(rect1.bottom, rect2.bottom) - max(rect1.top, rect2.top))

        val rect1Area = (rect1.right - rect1.left) * (rect1.bottom - rect1.top)
        val rect2Area = (rect2.right - rect2.left) * (rect2.bottom - rect2.top)

        val unionArea = rect1Area + rect2Area - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    data class Detection(val rect: RectF, val confidence: Float)

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Log.e("MainActivity", "Camera permissions not granted")
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
        cameraExecutor.shutdown()
    }
}
