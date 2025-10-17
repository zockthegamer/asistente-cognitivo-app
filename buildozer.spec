[app]
title = Asistente Cognitivo
package.name = asistente
package.domain = org.test
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,ttf
version = 0.1
requirements = -r requirements.txt
orientation = portrait
android.permissions = CAMERA, RECORD_AUDIO, INTERNET
android.archs = armeabi-v7a, arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
