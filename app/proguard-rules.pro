# MediaPipe Tasks binds its graph through JNI and reflects over its AutoValue containers, so the
# whole package has to survive shrinking. Release builds keep minification off today; these rules
# are here so turning it on later does not break the landmarker.
-keep class com.google.mediapipe.** { *; }
-keepclassmembers class com.google.mediapipe.** { *; }
-dontwarn com.google.mediapipe.**

# OpenGL entry points are resolved from native code.
-keep class javax.microedition.khronos.** { *; }
-dontwarn javax.microedition.khronos.**
