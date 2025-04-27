import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:deepfake/components/my_button.dart';
import 'package:deepfake/components/my_textfield.dart';
import 'package:deepfake/components/square_tile.dart';
import 'package:deepfake/pages/register_page.dart';
import 'package:deepfake/pages/auth_service.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  _LoginPageState createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();
  final AuthService _authService = AuthService();
  bool isLoading = false;
  bool isResetting = false;
  bool _obscurePassword = true;

  void showErrorDialog(String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title, style: const TextStyle(fontFamily: 'Poppins')),
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

  void signInWithEmailAndPassword() async {
    String? emailError = _authService.validateEmail(emailController.text.trim());
    if (emailError != null) {
      showErrorDialog("Invalid Email", emailError);
      return;
    }
    if (passwordController.text.trim().isEmpty) {
      showErrorDialog("Invalid Password", "Password cannot be empty");
      return;
    }

    setState(() {
      isLoading = true;
    });

    try {
      await _authService.signInWithEmailAndPassword(
        emailController.text.trim(),
        passwordController.text.trim(),
      );
    } on FirebaseAuthException catch (e) {
      setState(() {
        isLoading = false;
      });
      String errorMessage;
      switch (e.code) {
        case 'user-not-found':
          errorMessage = "No user found with this email.";
          break;
        case 'wrong-password':
          errorMessage = "Incorrect password. Please try again.";
          break;
        case 'invalid-email':
          errorMessage = "Invalid email address.";
          break;
        case 'invalid-credential':
          errorMessage = "The email or password is incorrect.";
          break;
        default:
          errorMessage = e.message ?? "An unknown error occurred.";
      }
      showErrorDialog("Sign-In Failed", errorMessage);
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  Future<void> signInWithGoogle() async {
    setState(() {
      isLoading = true;
    });

    try {
      await _authService.signInWithGoogle();
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      showErrorDialog("Google Sign-In Failed", e.toString());
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  void resetPassword() async {
    String? emailError = _authService.validateEmail(emailController.text.trim());
    if (emailError != null) {
      showErrorDialog("Invalid Email", emailError);
      return;
    }

    setState(() {
      isResetting = true;
    });
    try {
      await FirebaseAuth.instance.sendPasswordResetEmail(email: emailController.text.trim());
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            "Reset link sent to ${emailController.text.trim()}. Check your inbox!",
            style: const TextStyle(fontFamily: 'Poppins'),
          ),
          backgroundColor: Colors.blue.shade700,
          duration: const Duration(seconds: 5),
        ),
      );
      emailController.clear();
    } catch (e) {
      showErrorDialog("Reset Failed", "Error sending reset email: $e");
    } finally {
      setState(() {
        isResetting = false;
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
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(height: 60),
                FadeInAnimation(
                  child: Image.asset(
                    'lib/images/logo_new blue.png',
                    height: 140,
                  ),
                ),
                const SizedBox(height: 30),
                FadeInAnimation(
                  delay: 200,
                  child: Text(
                    'Welcome to Trustify',
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
                    controller: emailController,
                    hintText: 'Email',
                    obscureText: false,
                    showVisibilityToggle: false,
                  ),
                ),
                const SizedBox(height: 20),
                FadeInAnimation(
                  delay: 600,
                  child: MyTextField(
                    controller: passwordController,
                    hintText: 'Password',
                    obscureText: _obscurePassword,
                    showVisibilityToggle: true,
                    onVisibilityToggle: _togglePasswordVisibility,
                  ),
                ),
                const SizedBox(height: 15),
                FadeInAnimation(
                  delay: 800,
                  child: GestureDetector(
                    onTap: isResetting ? null : resetPassword,
                    child: Text(
                      'Forgot Password?',
                      style: TextStyle(
                        color: isResetting ? Colors.grey : Colors.blue.shade700,
                        fontWeight: FontWeight.w600,
                        fontFamily: 'Poppins',
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 30),
                FadeInAnimation(
                  delay: 1000,
                  child: isLoading
                      ? const CircularProgressIndicator(color: Colors.blue)
                      : MyButton(
                          onTap: signInWithEmailAndPassword,
                          text: "Sign In",
                        ),
                ),
                const SizedBox(height: 40),
                FadeInAnimation(
                  delay: 1200,
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 25.0),
                    child: Row(
                      children: [
                        Expanded(
                          child: Divider(thickness: 1, color: Colors.grey.shade300),
                        ),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 10.0),
                          child: Text(
                            'Or continue with',
                            style: TextStyle(
                              color: Colors.grey.shade600,
                              fontFamily: 'Poppins',
                            ),
                          ),
                        ),
                        Expanded(
                          child: Divider(thickness: 1, color: Colors.grey.shade300),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 30),
                FadeInAnimation(
                  delay: 1400,
                  child: GestureDetector(
                    onTap: signInWithGoogle,
                    child: const SquareTile(imagePath: 'lib/images/google.png'),
                  ),
                ),
                const SizedBox(height: 40),
                FadeInAnimation(
                  delay: 1600,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'Not a member?',
                        style: TextStyle(
                          color: Colors.grey.shade700,
                          fontFamily: 'Poppins',
                        ),
                      ),
                      const SizedBox(width: 4),
                      GestureDetector(
                        onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(builder: (context) => const RegisterPage()),
                        ),
                        child: Text(
                          'Register now',
                          style: TextStyle(
                            color: Colors.blue.shade700,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'Poppins',
                          ),
                        ),
                      ),
                    ],
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
    emailController.dispose();
    passwordController.dispose();
    super.dispose();
  }
}

class FadeInAnimation extends StatefulWidget {
  final Widget child;
  final int delay;

  const FadeInAnimation({required this.child, this.delay = 0, super.key});

  @override
  _FadeInAnimationState createState() => _FadeInAnimationState();
}

class _FadeInAnimationState extends State<FadeInAnimation> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );
    _animation = CurvedAnimation(parent: _controller, curve: Curves.easeIn);
    Future.delayed(Duration(milliseconds: widget.delay), () {
      if (mounted) _controller.forward();
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _animation,
      child: widget.child,
    );
  }
}