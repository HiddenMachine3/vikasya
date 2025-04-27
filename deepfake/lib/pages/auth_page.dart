import 'package:deepfake/pages/home_page.dart';
import 'package:deepfake/pages/login_page.dart';
import 'package:deepfake/pages/verification_page.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

class AuthPage extends StatelessWidget {
  const AuthPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: StreamBuilder<User?>(
        stream: FirebaseAuth.instance.authStateChanges(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator(color: Colors.blue));
          }
          if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}', style: const TextStyle(fontFamily: 'Poppins')));
          }
          if (snapshot.hasData) {
            final user = snapshot.data!;
            if (user.emailVerified) {
              return const HomePage();
            } else {
              return const VerificationPage();
            }
          }
          return const LoginPage();
        },
      ),
    );
  }
}