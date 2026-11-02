import java.net.URL

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// ---------------------------------------------------------------------------
// Model assets
//
// The two MediaPipe assets are large binaries and are deliberately NOT checked
// in. `downloadModelAssets` fetches them on the first build and then becomes a
// no-op. Every build verifies that both files are present (and that the OBJ
// actually contains the 468-vertex / 898-triangle canonical mesh) so a missing
// or truncated asset fails loudly instead of producing an APK that crashes at
// runtime with "Failed to create FaceLandmarker".
// ---------------------------------------------------------------------------
val assetsDir = layout.projectDirectory.dir("src/main/assets")
val faceLandmarkerAsset = assetsDir.file("face_landmarker.task")
val canonicalMeshAsset = assetsDir.file("canonical_face_model.obj")

val faceLandmarkerUrl =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
val canonicalMeshUrl =
    "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj"

val downloadModelAssets = tasks.register("downloadModelAssets") {
    group = "facepuppet"
    description = "Downloads the Face Landmarker model and the canonical face mesh into src/main/assets."

    doLast {
        fun fetch(url: String, target: java.io.File, minBytes: Long, what: String) {
            if (target.isFile && target.length() >= minBytes) {
                logger.lifecycle("FacePuppet: $what already present (${target.length()} bytes)")
                return
            }
            val tmp = java.io.File(target.parentFile, target.name + ".part")
            target.parentFile.mkdirs()
            logger.lifecycle("FacePuppet: downloading $what -> ${target.absolutePath}")
            try {
                val connection = URL(url).openConnection()
                connection.connectTimeout = 30_000
                connection.readTimeout = 120_000
                connection.setRequestProperty("User-Agent", "FacePuppet-Gradle")
                connection.getInputStream().use { input ->
                    tmp.outputStream().use { output -> input.copyTo(output) }
                }
            } catch (error: Exception) {
                tmp.delete()
                throw GradleException(
                    """
                    |FacePuppet: could not download $what.
                    |  url    : $url
                    |  reason : ${error.message}
                    |  expect : at least ${minBytes / 1024} KiB
                    |
                    |Workaround: download the file yourself with any browser/curl and place it at
                    |  ${target.absolutePath}
                    |then re-run the build.
                    """.trimMargin(),
                    error,
                )
            }
            if (tmp.length() < minBytes) {
                val size = tmp.length()
                tmp.delete()
                throw GradleException(
                    "FacePuppet: downloaded $what is only $size bytes (expected >= ${minBytes / 1024} KiB). " +
                        "The download was truncated or blocked by a proxy; place the file manually at " +
                        target.absolutePath,
                )
            }
            if (target.exists()) target.delete()
            check(tmp.renameTo(target)) { "FacePuppet: could not move ${tmp.name} to ${target.absolutePath}" }
            logger.lifecycle("FacePuppet: $what ready (${target.length()} bytes)")
        }

        fetch(faceLandmarkerUrl, faceLandmarkerAsset.asFile, 1_000_000L, "face_landmarker.task")
        fetch(canonicalMeshUrl, canonicalMeshAsset.asFile, 30_000L, "canonical_face_model.obj")

        val obj = canonicalMeshAsset.asFile
        val vertexCount = obj.useLines { lines -> lines.count { it.startsWith("v ") } }
        val faceCount = obj.useLines { lines -> lines.count { it.startsWith("f ") } }
        if (vertexCount != 468 || faceCount != 898) {
            throw GradleException(
                "FacePuppet: ${obj.name} is not the canonical 468-vertex mesh " +
                    "(found $vertexCount vertices / $faceCount triangles, expected 468 / 898). " +
                    "Delete the file and rebuild so it is downloaded again.",
            )
        }
    }
}

// Cheap presence check that runs on every build, even when the download task is up to date.
val verifyModelAssets = tasks.register("verifyModelAssets") {
    group = "facepuppet"
    description = "Fails the build when the MediaPipe assets are missing."
    doLast {
        val missing = listOf(faceLandmarkerAsset.asFile, canonicalMeshAsset.asFile).filterNot { it.isFile }
        if (missing.isNotEmpty()) {
            throw GradleException(
                "FacePuppet: missing model asset(s):\n" +
                    missing.joinToString("\n") { "  ${it.absolutePath}" } +
                    "\nRun `./gradlew downloadModelAssets` (needs network access) or copy the files in manually.",
            )
        }
    }
}

tasks.named("preBuild") { dependsOn(downloadModelAssets) }
tasks.named("preBuild") { dependsOn(verifyModelAssets) }

android {
    namespace = "com.facepuppet"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.facepuppet"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    androidResources {
        // MediaPipe maps the .task bundle and the canonical OBJ straight out of the APK;
        // both MUST stay uncompressed or AssetManager.openFd() fails on device.
        noCompress.addAll(listOf("task", "tflite", "obj"))
    }

    packaging {
        resources.excludes += setOf("/META-INF/{AL2.0,LGPL2.1}", "/META-INF/DEPENDENCIES")
    }

    buildFeatures {
        buildConfig = true
    }
}

dependencies {
    // On-device face landmarker (blendshapes + transformation matrices), CPU or GPU delegate.
    implementation("com.google.mediapipe:tasks-vision:0.10.14")

    val cameraX = "1.4.0"
    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")

    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
