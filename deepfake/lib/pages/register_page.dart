import 'package:deepfake/pages/login_page.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:deepfake/pages/auth_service.dart';
import 'package:deepfake/components/my_textfield.dart';
import 'package:deepfake/components/my_button.dart';

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key});

  @override
  _RegisterPageState createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final TextEditingController nameController = TextEditingController();
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();
  final AuthService _authService = AuthService();
  bool isLoading = false;
  bool _obscurePassword = true;

  void showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Registration Failed", style: TextStyle(fontFamily: 'Poppins')),
        content: Text(message, style: const TextStyle(fontFamily: 'Poppins')),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("OK", style: TextStyle(fontFamily: 'Poppins')),
          ),
        ],
      ),
    );
  }

  void registerUser() async {
    String? emailError = _authService.validateEmail(emailController.text.trim());
    if (emailError != null) {
      showErrorDialog(emailError);
      return;
    }
    if (passwordController.text.trim().length < 6) {
      showErrorDialog("Password must be at least 6 characters long");
      return;
    }

    setState(() {
      isLoading = true;
    });

    try {
      await _authService.registerWithEmailAndPassword(
        emailController.text.trim(),
        passwordController.text.trim(),
      );
      Navigator.pop(context);
    } on FirebaseAuthException catch (e) {
      setState(() {
        isLoading = false;
      });
      showErrorDialog(e.message ?? "An unknown error occurred.");
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  void _togglePasswordVisibility() {
    setState(() {
      _obscurePassword = !_obscurePassword;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white, // changed to white background
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () {
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(builder: (context) => const LoginPage()),
            );
          },
        ),
      ),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(height: 20),
                FadeInAnimation(
                  child: Icon(Icons.person_add, size: 100, color: Colors.blue.shade700),
                ),
                const SizedBox(height: 30),
                FadeInAnimation(
                  delay: 200,
                  child: Text(
                    'Create Account',
                    style: TextStyle(
                      color: Colors.blue.shade900,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'Poppins',
                    ),
                  ),
                ),
                const SizedBox(height: 40),
                FadeInAnimation(
                  delay: 400,
                  child: MyTextField(
                    controller: nameController,
                    hintText: "Full Name",
                    obscureText: false,
                    showVisibilityToggle: false,
                  ),
                ),
                const SizedBox(height: 20),
                FadeInAnimation(
                  delay: 600,
                  child: MyTextField(
                    controller: emailController,
                    hintText: "Email",
                    obscureText: false,
                    showVisibilityToggle: false,
                  ),
                ),
                const SizedBox(height: 20),
                FadeInAnimation(
                  delay: 800,
                  child: MyTextField(
                    controller: passwordController,
                    hintText: "Password",
                    obscureText: _obscurePassword,
                    showVisibilityToggle: true,
                    onVisibilityToggle: _togglePasswordVisibility,
                  ),
                ),
                const SizedBox(height: 30),
                FadeInAnimation(
                  delay: 1000,
                  child: isLoading
                      ? const CircularProgressIndicator(color: Colors.blue)
                      : MyButton(
                          onTap: registerUser,
                          text: "Register",
                        ),
                ),
                const SizedBox(height: 20),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    nameController.dispose();
    emailController.dispose();
    passwordController.dispose();
    super.dispose();
  }
}