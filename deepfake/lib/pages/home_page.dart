import 'package:deepfake/pages/login_page.dart';
import 'package:flutter/material.dart';
import 'package:deepfake/pages/auth_service.dart';
import 'package:deepfake/components/my_button.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:deepfake/pages/file_picker_page.dart'; // Import the FilePickerPage

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final AuthService _authService = AuthService();
  bool isLoading = false;

  @override
  void initState() {
    super.initState();
    _checkFirstLogin();
  }

  Future<void> _checkFirstLogin() async {
    final prefs = await SharedPreferences.getInstance();
    final isFirstLogin = prefs.getBool('isFirstLogin') ?? true;
    final user = _authService.getCurrentUser();

    if (isFirstLogin && user != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Welcome, ${user.email}! You’ve successfully logged in.',
            style: const TextStyle(fontFamily: 'Poppins'),
          ),
          backgroundColor: Colors.blue.shade700,
          duration: const Duration(seconds: 5),
        ),
      );
      await prefs.setBool('isFirstLogin', false);
    }
  }

  void signOut() async {
    setState(() {
      isLoading = true;
    });
    try {
      await _authService.signOut();
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool('isFirstLogin', true);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error signing out: $e', style: const TextStyle(fontFamily: 'Poppins')),
          backgroundColor: Colors.red.shade700,
        ),
      );
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  void navigateToFilePicker() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const FilePickerPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final user = _authService.getCurrentUser();
    return Scaffold(
      body: Container(
        color: Colors.white,
        child: SafeArea(
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                FadeInAnimation(
                  child: Image.asset(
                    'lib/images/logo_new blue.png',
                    height: 100,
                  ),
                ),
                const SizedBox(height: 20),
                FadeInAnimation(
                  child: Icon(Icons.home, size: 100, color: Colors.blue.shade700),
                ),
                const SizedBox(height: 30),
                FadeInAnimation(
                  delay: 200,
                  child: Text(
                    'Welcome, ${user?.email ?? 'User'}!',
                    style: TextStyle(
                      color: Colors.blue.shade900,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'Poppins',
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
                const SizedBox(height: 40),
                FadeInAnimation(
                  delay: 400,
                  child: isLoading
                      ? const CircularProgressIndicator(color: Colors.blue)
                      : MyButton(
                          onTap: signOut,
                          text: 'Sign Out',
                        ),
                ),
                const SizedBox(height: 20), // Add spacing between buttons
                FadeInAnimation(
                  delay: 600,
                  child: MyButton(
                    onTap: navigateToFilePicker,
                    text: 'Select File',
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