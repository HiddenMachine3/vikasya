plugins {
    id("com.android.application")
    id("kotlin-android")
    id("dev.flutter.flutter-gradle-plugin") // ⚡ Flutter plugin AFTER Android and Kotlin
}

android {
    namespace = "com.example.deepfake"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = "27.0.12077973" // ✅ Manually set NDK version

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString() // Ensure compatibility with Java 17
    }

    // Ensure Gradle uses Java 17 for both Java and Kotlin compilation
    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of(17)) // Set to Java 17
        }
    }

    defaultConfig {
        applicationId = "com.example.distinkt"
        minSdk = 24 // ✅ Manually set minSdk to 24 for Amplify
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}