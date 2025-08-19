package com.example.realtime_object

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var boundingBoxes: List<RectF> = emptyList()
    private var labels: List<String> = emptyList()
    private var scores: List<Float> = emptyList()

    // Optimized Paint objects
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 8f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val textBackgroundPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        alpha = 180
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 42f
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }

    fun setResults(boxes: List<RectF>, labels: List<String>, scores: List<Float>) {
        this.boundingBoxes = boxes
        this.labels = labels
        this.scores = scores
        invalidate() // Trigger redraw
    }

    fun clearResults() {
        boundingBoxes = emptyList()
        labels = emptyList()
        scores = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (i in boundingBoxes.indices) {
            val rect = boundingBoxes[i]
            val label = if (i < labels.size) labels[i] else "Object"
            val score = if (i < scores.size) scores[i] else 0f

            // Draw bounding box
            canvas.drawRect(rect, boxPaint)

            // Prepare label text
            val text = "$label: ${"%.1f%%".format(score * 100)}"
            val textBounds = Rect()
            textPaint.getTextBounds(text, 0, text.length, textBounds)

            // Calculate text background position
            val textX = rect.left
            val textY = rect.top - 10f
            val backgroundRect = RectF(
                textX,
                textY - textBounds.height() - 10f,
                textX + textBounds.width() + 20f,
                textY + 5f
            )

            // Draw text background and text
            canvas.drawRoundRect(backgroundRect, 8f, 8f, textBackgroundPaint)
            canvas.drawText(text, textX + 10f, textY - 5f, textPaint)
        }
    }
}
