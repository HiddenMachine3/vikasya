import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:deepfake/pages/auth_service.dart';
import 'package:deepfake/pages/login_page.dart';
import 'package:deepfake/components/my_button.dart';

class VerificationPage extends StatefulWidget {
  const VerificationPage({super.key});

  @override
  _VerificationPageState createState() => _VerificationPageState();
}

class _VerificationPageState extends State<VerificationPage> {
  final AuthService _authService = AuthService();
  bool isLoading = false;

  Future<void> resendVerificationEmail() async {
    setState(() {
      isLoading = true;
    });
    try {
      final user = _authService.getCurrentUser();
      if (user != null && !user.emailVerified) {
        await user.sendEmailVerification();
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Verification email resent to ${user.email}',
              style: const TextStyle(fontFamily: 'Poppins'),
            ),
            backgroundColor: Colors.blue.shade700,
          ),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $e', style: const TextStyle(fontFamily: 'Poppins')),
          backgroundColor: Colors.red.shade700,
        ),
      );
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  Future<void> checkEmailVerified() async {
    setState(() {
      isLoading = true;
    });
    try {
      final user = _authService.getCurrentUser();
      if (user != null) {
        await user.reload();
        if (user.emailVerified) {
          await _authService.signOut();
          Navigator.pushAndRemoveUntil(
            context,
            MaterialPageRoute(builder: (context) => const LoginPage()),
            (route) => false,
          );
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: const Text('Email verified! Please sign in.', style: TextStyle(fontFamily: 'Poppins')),
              backgroundColor: Colors.blue.shade700,
            ),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: const Text('Email not yet verified.', style: TextStyle(fontFamily: 'Poppins')),
              backgroundColor: Colors.orange.shade700,
            ),
          );
        }
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $e', style: const TextStyle(fontFamily: 'Poppins')),
          backgroundColor: Colors.red.shade700,
        ),
      );
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  Widget customOutlinedButton({required String text, required VoidCallback onPressed}) {
    return OutlinedButton(
      onPressed: onPressed,
      style: OutlinedButton.styleFrom(
        side: BorderSide(color: Colors.blue.shade700),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: Colors.blue.shade700,
          fontSize: 16,
          fontFamily: 'Poppins',
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final user = _authService.getCurrentUser();
    return Scaffold(
      backgroundColor: Colors.white, // White background
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(height: 40),
                FadeInAnimation(
                  child: Icon(Icons.email, size: 100, color: Colors.blue.shade700),
                ),
                const SizedBox(height: 30),
                FadeInAnimation(
                  delay: 200,
                  child: Text(
                    'Verify Your Email',
                    style: TextStyle(
                      color: Colors.blue.shade900,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'Poppins',
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                FadeInAnimation(
                  delay: 400,
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Text(
                      'A verification email has been sent to ${user?.email ?? 'your email'}. Check your inbox (and spam folder).',
                      textAlign: TextAlign.center,
                      style: const TextStyle(fontSize: 16, fontFamily: 'Poppins'),
                    ),
                  ),
                ),
                const SizedBox(height: 40),
                FadeInAnimation(
                  delay: 600,
                  child: isLoading
                      ? const CircularProgressIndicator(color: Colors.blue)
                      : Column(
                          children: [
                            customOutlinedButton(
                              onPressed: resendVerificationEmail,
                              text: 'Resend Email',
                            ),
                            const SizedBox(height: 20),
                            customOutlinedButton(
                              onPressed: checkEmailVerified,
                              text: 'I’ve Verified',
                            ),
                          ],
                        ),
                ),
                const SizedBox(height: 30),
                FadeInAnimation(
                  delay: 800,
                  child: TextButton(
                    onPressed: () async => await _authService.signOut(),
                    child: Text(
                      'Sign Out',
                      style: TextStyle(
                        color: Colors.blue.shade700,
                        fontWeight: FontWeight.bold,
                        fontFamily: 'Poppins',
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}