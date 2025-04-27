import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path/path.dart' as path;
import 'package:just_audio/just_audio.dart';
import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:http_parser/http_parser.dart';

class FilePickerPage extends StatefulWidget {
  const FilePickerPage({super.key});

  @override
  _FilePickerPageState createState() => _FilePickerPageState();
}

class _FilePickerPageState extends State<FilePickerPage> {
  String? _filePath;
  bool _isLoading = false;
  String? _detectionResult;
  final player = AudioPlayer();
  bool _isImage = false;

  final List<String> _audioExtensions = ['mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac'];
  final List<String> _imageExtensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif'];

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    await [
      Permission.storage,
      Permission.microphone,
    ].request();
  }

  Future<void> _pickFile() async {
    setState(() {
      _isLoading = true;
      _detectionResult = null;
    });

    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        allowMultiple: false,
        type: FileType.custom,
        allowedExtensions: [..._audioExtensions, ..._imageExtensions],
      );

      if (result != null && result.files.isNotEmpty) {
        String? pickedPath = result.files.single.path;
        String extension = path.extension(pickedPath!).toLowerCase();

        bool isImage = _imageExtensions.contains(extension.replaceAll('.', ''));
        bool isAudio = _audioExtensions.contains(extension.replaceAll('.', ''));

        setState(() {
          _filePath = pickedPath;
          _isImage = isImage;
        });

        if (isAudio) {
          await player.setFilePath(_filePath!);
          player.play();
        }

        if (_filePath != null) {
          await _runInference(_filePath!);
        }

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                'Selected ${isImage ? 'image' : 'audio'}: ${path.basename(_filePath!)}',
                style: const TextStyle(fontFamily: 'Poppins'),
              ),
              backgroundColor: Colors.blue.shade700,
            ),
          );
        }
      }
    } catch (e) {
      print('Error picking file: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error picking file: $e', style: const TextStyle(fontFamily: 'Poppins')),
            backgroundColor: Colors.red.shade700,
          ),
        );
      }
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _runInference(String filePath) async {
    if (_filePath == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text('No file selected'),
            backgroundColor: Colors.red.shade700,
          ),
        );
      }
      return;
    }

    try {
      File file = File(filePath);
      String extension = path.extension(filePath).toLowerCase().replaceAll('.', '');

      String mimeType = _imageExtensions.contains(extension) ? 'image' : 'audio';
      String subtype = extension == 'jpg' ? 'jpeg' : extension;

      var uri = Uri.parse('http://192.168.233.119:9000/analyze');
      var request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath(
          'file',
          filePath,
          contentType: MediaType(mimeType, subtype),
        ));

      var response = await request.send();

      if (response.statusCode == 200) {
        String responseBody = await response.stream.bytesToString();
        Map<String, dynamic> responseJson = jsonDecode(responseBody);

        setState(() {
          _detectionResult = const JsonEncoder.withIndent('  ').convert(responseJson);
        });

        if (responseJson.containsKey('prediction')) {
          _showPredictionDialog(responseJson['prediction'].toString());
        }
      } else {
        print('Failed to get response: ${response.statusCode}');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: const Text('Error connecting to server'),
              backgroundColor: Colors.red.shade700,
            ),
          );
        }
      }
    } catch (e) {
      print('Error running inference: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.red.shade700,
          ),
        );
      }
    }
  }

  void _showPredictionDialog(String prediction) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Prediction Result', style: TextStyle(fontFamily: 'Poppins', fontWeight: FontWeight.bold)),
        content: Text(prediction, style: const TextStyle(fontFamily: 'Poppins', fontSize: 16)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK', style: TextStyle(fontFamily: 'Poppins', fontWeight: FontWeight.bold)),
          ),
        ],
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      ),
    );
  }

  @override
  void dispose() {
    player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(
        title: const Text('Deep Fake Detector', style: TextStyle(fontFamily: 'Poppins')),
        backgroundColor: Colors.blue.shade700,
        elevation: 5,
        centerTitle: true,
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [BoxShadow(color: Colors.grey.shade300, blurRadius: 10, offset: const Offset(0, 4))],
                ),
                child: Column(
                  children: [
                    Image.asset('lib/images/logo_new blue.png', height: 100),
                    const SizedBox(height: 20),
                    Text(
                      _filePath == null ? 'No file selected' : path.basename(_filePath!),
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, fontFamily: 'Poppins', color: Colors.blue.shade900),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 10),
                    if (_filePath != null)
                      Text(
                        'Type: ${_isImage ? 'Image' : 'Audio'}',
                        style: TextStyle(fontSize: 16, color: Colors.blue.shade700, fontFamily: 'Poppins'),
                      ),
                    const SizedBox(height: 20),
                    if (_isImage && _filePath != null)
                      ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: Image.file(
                          File(_filePath!),
                          height: 200,
                          width: double.infinity,
                          fit: BoxFit.cover,
                        ),
                      ),
                    const SizedBox(height: 20),
                    _isLoading
                        ? const CircularProgressIndicator(color: Colors.blue)
                        : ElevatedButton(
                            onPressed: _pickFile,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.blue.shade700,
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                              padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                            ),
                            child: const Text(
                              'Select File (Audio / Image)',
                              style: TextStyle(
                                fontSize: 16,
                                fontFamily: 'Poppins',
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                  ],
                ),
              ),
              if (_detectionResult != null) ...[
                const SizedBox(height: 30),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [BoxShadow(color: Colors.grey.shade300, blurRadius: 10, offset: const Offset(0, 4))],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Full JSON Response:', style: TextStyle(fontSize: 18, fontFamily: 'Poppins', fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),
                      SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: Text(
                          _detectionResult!,
                          style: const TextStyle(
                            fontFamily: 'Courier',
                            fontSize: 14,
                            color: Colors.black87,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}