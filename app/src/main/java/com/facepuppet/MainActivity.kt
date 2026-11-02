package com.facepuppet

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import com.facepuppet.camera.CameraController
import com.facepuppet.face.FaceLandmarkerEngine
import com.facepuppet.io.PhotoIngest
import com.facepuppet.io.Snapshot
import com.facepuppet.io.StencilConfigChooser
import com.facepuppet.render.PuppetRenderer
import com.facepuppet.render.PuppetState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Single-activity host: permission flow, Photo Picker, the GL puppet preview and the transport
 * controls (record / snapshot / recenter).
 *
 * Rotation: the activity is portrait-locked and handles the configuration changes itself, so the GL
 * context and the button state survive a rotation without a recreate. Recording is always stopped
 * in `onPause`, which is also where the MP4 gets finalised.
 */
class MainActivity : ComponentActivity() {

    private lateinit var state: PuppetState
    private lateinit var renderer: PuppetRenderer
    private lateinit var glView: GLSurfaceView
    private lateinit var configChooser: StencilConfigChooser
    private lateinit var camera: CameraController

    private var engine: FaceLandmarkerEngine? = null
    private var engineLoading = false

    private lateinit var hintView: TextView
    private lateinit var trackHintView: TextView
    private lateinit var permissionPanel: LinearLayout
    private lateinit var recordButton: TextView
    private lateinit var snapshotButton: TextView
    private lateinit var recenterButton: TextView
    private lateinit var fpsBadge: TextView
    private lateinit var prefs: SharedPreferences

    private var cameraGranted = false
    private var recordingActive = false
    private var recordStartPending = false

    /** False only when the canonical mesh asset is missing/corrupt: the UI still opens, no GL. */
    private var glReady = false

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        cameraGranted = granted
        if (granted) {
            permissionPanel.visibility = View.GONE
            startPipeline()
        } else {
            showPermissionPanel()
        }
    }

    private val pickPhoto = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) ingestPhoto(uri)
    }

    private val uiTicker = object : Runnable {
        override fun run() {
            val fps = state.fps
            if (BuildConfig.DEBUG) {
                fpsBadge.text = getString(R.string.fps_format, fps)
                fpsBadge.visibility = View.VISIBLE
            } else {
                fpsBadge.visibility = View.GONE
            }
            val tracking = state.latestFrame.get().hasFace
            trackHintView.visibility =
                if (state.rig.get() != null && !tracking) View.VISIBLE else View.GONE
            glView.postDelayed(this, 1000L)
        }
    }

    // =============================================================================================
    // Lifecycle
    // =============================================================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        state = PuppetState()

        configChooser = StencilConfigChooser()
        glView = GLSurfaceView(this).apply {
            setEGLContextClientVersion(3)
            setEGLConfigChooser(configChooser)
            preserveEGLContextOnPause = true
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        val createdRenderer = try {
            PuppetRenderer(this, state)
        } catch (error: Throwable) {
            Log.e(TAG, "could not create the puppet renderer", error)
            null
        }
        if (createdRenderer == null) {
            showMessage(getString(R.string.error_model))
        } else {
            renderer = createdRenderer
            glView.setRenderer(createdRenderer)
            glReady = true
        }

        state.eglConfigProvider = { configChooser.config }
        state.snapshotDelivery = { pixels, width, height ->
            Snapshot.save(this, pixels, width, height) { label -> onSnapshotSaved(label) }
        }
        state.recordingStateListener = { active, label ->
            runOnUiThread { onRecordingStateChanged(active, label) }
        }

        buildUi()
        setContentView(rootView)

        camera = CameraController(
            context = this,
            onUprightFrame = { bitmap, timestampMs ->
                engine?.submitBitmap(bitmap, timestampMs)
            },
            onError = { message -> runOnUiThread { showMessage(message) } },
        )

        cameraGranted = checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (cameraGranted) {
            startPipeline()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }

        if (!prefs.getBoolean(KEY_HINT_SEEN, false)) hintView.visibility = View.VISIBLE
    }

    override fun onResume() {
        super.onResume()
        hideSystemBars()
        glView.onResume()
        glView.post(uiTicker)
        if (cameraGranted) startPipeline()
    }

    override fun onPause() {
        // Stop the capture pipeline *and* finalise any recording before the GL context goes away.
        camera.stop()
        stopRecordingBlocking()
        glView.removeCallbacks(uiTicker)
        glView.onPause()
        releaseEngine()
        super.onPause()
    }

    override fun onDestroy() {
        camera.shutdown()
        releaseEngine()
        if (glReady) glView.queueEvent { renderer.shutdown() }
        super.onDestroy()
    }

    // =============================================================================================
    // Pipeline
    // =============================================================================================

    private fun startPipeline() {
        if (engineLoading || engine != null) {
            if (engine != null && !camera.isRunning) camera.start(this)
            return
        }
        engineLoading = true
        lifecycleScope.launch {
            val created = withContext(Dispatchers.IO) {
                FaceLandmarkerEngine(applicationContext).apply {
                    onFrame = { frame -> state.latestFrame.set(frame) }
                    onError = { message -> runOnUiThread { Log.w(TAG, message) } }
                    initialize()
                }
            }
            engineLoading = false
            if (created.delegateName == "none") {
                created.release()
                showMessage(getString(R.string.error_model))
                return@launch
            }
            engine = created
            Log.i(TAG, "landmarker delegate: ${created.delegateName}")
            if (cameraGranted) camera.start(this@MainActivity)
        }
    }

    private fun releaseEngine() {
        engine?.release()
        engine = null
        engineLoading = false
    }

    // =============================================================================================
    // Photo
    // =============================================================================================

    private fun launchPhotoPicker() {
        pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }

    private fun ingestPhoto(uri: Uri) {
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) { PhotoIngest.ingest(applicationContext, uri) }
            when (result) {
                is PhotoIngest.Result.Success -> {
                    state.rig.set(result.rig)
                    hintView.visibility = View.GONE
                    prefs.edit().putBoolean(KEY_HINT_SEEN, true).apply()
                }

                PhotoIngest.Result.NoFace -> showMessage(getString(R.string.error_no_face_in_photo))
                is PhotoIngest.Result.Failure ->
                    showMessage(result.message.ifBlank { getString(R.string.error_photo_load) })
            }
        }
    }

    // =============================================================================================
    // Transport
    // =============================================================================================

    private fun toggleRecording() {
        if (!glReady) return
        if (!recordingActive) {
            recordStartPending = true
            glView.queueEvent { renderer.startRecording() }
        } else {
            glView.queueEvent { renderer.stopRecording() }
        }
    }

    private fun stopRecordingBlocking() {
        if (!glReady || !recordingActive) return
        val latch = CountDownLatch(1)
        glView.queueEvent {
            try {
                renderer.stopRecording()
            } finally {
                latch.countDown()
            }
        }
        // Finalising an MP4 takes a few milliseconds; the timeout only guards a dead GL thread.
        try {
            if (!latch.await(800, TimeUnit.MILLISECONDS)) Log.w(TAG, "record stop timed out")
        } catch (interrupted: InterruptedException) {
            Thread.currentThread().interrupt()
        }
    }

    private fun onRecordingStateChanged(active: Boolean, label: String?) {
        val startFailed = !active && recordStartPending
        recordingActive = active
        recordStartPending = false
        recordButton.text = getString(if (active) R.string.stop else R.string.record)
        recordButton.setCompoundDrawablesWithIntrinsicBounds(
            if (active) R.drawable.ic_record_square else R.drawable.ic_record_dot,
            0,
            0,
            0,
        )
        recordButton.setBackgroundResource(if (active) R.drawable.bg_pill_active else R.drawable.bg_pill)
        if (active) {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else {
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            when {
                label != null -> showMessage(getString(R.string.saved_video, label))
                startFailed -> showMessage(getString(R.string.error_record_start))
                else -> {
                    // A null label after a stop means nothing usable was produced.
                    Log.w(TAG, "recording produced no file")
                    showMessage(getString(R.string.error_save))
                }
            }
        }
    }

    private fun takeSnapshot() {
        if (!glReady) return
        state.snapshotRequested.set(true)
    }

    private fun onSnapshotSaved(label: String?) {
        if (label == null) {
            showMessage(getString(R.string.error_save))
        } else {
            showMessage(getString(R.string.saved_snapshot, label))
        }
    }

    // =============================================================================================
    // UI
    // =============================================================================================

    private lateinit var rootView: FrameLayout

    private fun buildUi() {
        rootView = FrameLayout(this).apply {
            setBackgroundColor(getColor(R.color.puppet_bg))
        }
        rootView.addView(
            glView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        // Full-bleed preview FAB, bottom-end, floating above the transport row.
        val fab = ImageButton(this).apply {
            setBackgroundResource(R.drawable.bg_fab)
            setImageResource(R.drawable.ic_camera)
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            setPadding(dp(16), dp(16), dp(16), dp(16))
            contentDescription = getString(R.string.a11y_choose_photo)
            setOnClickListener { launchPhotoPicker() }
        }
        rootView.addView(
            fab,
            FrameLayout.LayoutParams(dp(64), dp(64), Gravity.BOTTOM or Gravity.END).apply {
                setMargins(0, 0, dp(20), dp(104))
            },
        )

        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        recordButton = makePill(R.drawable.ic_record_dot, getString(R.string.record), R.drawable.bg_pill).apply {
            contentDescription = getString(R.string.a11y_record)
            setOnClickListener { toggleRecording() }
        }
        snapshotButton = makePill(R.drawable.ic_snapshot, getString(R.string.snapshot), R.drawable.bg_pill).apply {
            contentDescription = getString(R.string.a11y_snapshot)
            setOnClickListener { takeSnapshot() }
        }
        recenterButton = makePill(R.drawable.ic_recenter, getString(R.string.recenter), R.drawable.bg_pill).apply {
            contentDescription = getString(R.string.a11y_recenter)
            setOnClickListener { state.recenterRequested.set(true) }
        }
        row.addView(recordButton, pillParams())
        row.addView(snapshotButton, pillParams())
        row.addView(recenterButton, pillParams())
        rootView.addView(
            row,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL,
            ).apply { setMargins(dp(12), 0, dp(12), dp(28)) },
        )

        hintView = TextView(this).apply {
            text = getString(R.string.hint_center_face)
            setTextColor(getColor(R.color.puppet_text))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
            gravity = Gravity.CENTER
            setPadding(dp(20), dp(14), dp(20), dp(14))
            setBackgroundResource(R.drawable.bg_pill)
            visibility = View.GONE
        }
        rootView.addView(
            hintView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.CENTER_HORIZONTAL,
            ).apply { setMargins(dp(24), dp(72), dp(24), 0) },
        )

        trackHintView = TextView(this).apply {
            text = getString(R.string.hint_no_face_live)
            setTextColor(getColor(R.color.puppet_text))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            gravity = Gravity.CENTER
            setPadding(dp(16), dp(10), dp(16), dp(10))
            setBackgroundResource(R.drawable.bg_pill)
            visibility = View.GONE
        }
        rootView.addView(
            trackHintView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.CENTER_HORIZONTAL,
            ).apply { setMargins(dp(24), dp(24), dp(24), 0) },
        )

        fpsBadge = TextView(this).apply {
            setTextColor(getColor(R.color.puppet_text))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            visibility = View.GONE
        }
        rootView.addView(
            fpsBadge,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.START,
            ).apply { setMargins(dp(16), dp(16), 0, 0) },
        )

        permissionPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(dp(24), dp(24), dp(24), dp(24))
            setBackgroundColor(getColor(R.color.puppet_scrim))
            visibility = View.GONE
        }
        val permissionText = TextView(this).apply {
            text = getString(R.string.error_camera_permission)
            setTextColor(getColor(R.color.puppet_text))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
            gravity = Gravity.CENTER
        }
        val retry = makePill(0, getString(R.string.permission_retry), R.drawable.bg_pill).apply {
            setOnClickListener { requestCamera.launch(Manifest.permission.CAMERA) }
        }
        val settings = makePill(0, getString(R.string.permission_settings), R.drawable.bg_pill).apply {
            setOnClickListener { openAppSettings() }
        }
        permissionPanel.addView(permissionText)
        permissionPanel.addView(retry, pillParams())
        permissionPanel.addView(settings, pillParams())
        rootView.addView(
            permissionPanel,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )
    }

    private fun makePill(iconRes: Int, label: String, backgroundRes: Int): TextView {
        return TextView(this).apply {
            text = label
            setTextColor(getColor(R.color.puppet_text))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            gravity = Gravity.CENTER
            setBackgroundResource(backgroundRes)
            setPadding(dp(16), dp(12), dp(18), dp(12))
            if (iconRes != 0) {
                setCompoundDrawablesWithIntrinsicBounds(iconRes, 0, 0, 0)
                compoundDrawablePadding = dp(8)
            }
            isClickable = true
            isFocusable = true
        }
    }

    private fun pillParams(): LinearLayout.LayoutParams = LinearLayout.LayoutParams(
        ViewGroup.LayoutParams.WRAP_CONTENT,
        ViewGroup.LayoutParams.WRAP_CONTENT,
    ).apply { setMargins(dp(4), 0, dp(4), 0) }

    private fun showPermissionPanel() {
        permissionPanel.visibility = View.VISIBLE
        hintView.visibility = View.GONE
    }

    private fun showMessage(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun openAppSettings() {
        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
            data = Uri.fromParts("package", packageName, null)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        try {
            startActivity(intent)
        } catch (error: Throwable) {
            Log.w(TAG, "cannot open settings", error)
            showMessage(getString(R.string.error_camera_permission))
        }
    }

    private fun hideSystemBars() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
            val controller = window.insetsController
            controller?.hide(WindowInsets.Type.systemBars())
            controller?.systemBarsBehavior =
                WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                    or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                    or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                    or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                    or View.SYSTEM_UI_FLAG_FULLSCREEN
                    or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                )
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            // `window.attributes` hands back a copy: mutate it and write it back, or the cutout
            // mode is silently dropped.
            val attributes = window.attributes
            attributes.layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            window.attributes = attributes
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    companion object {
        private const val TAG = "MainActivity"
        private const val PREFS = "facepuppet"
        private const val KEY_HINT_SEEN = "hint_seen"
    }
}
