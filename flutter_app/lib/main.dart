import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(FacePalsyApp());

class FacePalsyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Palsy Detector',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _image;
  String _result = '';
  bool _loading = false;

  Future<void> _captureImage() async {
    final XFile? photo = await _picker.pickImage(source: ImageSource.camera);
    if (photo != null) {
      setState(() {
        _image = File(photo.path);
        _result = '';
      });
    }
  }

  Future<void> _analyzeImage() async {
    if (_image == null) return;

    setState(() {
      _loading = true;
      _result = '';
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('https://stroke-detection-1oa4.onrender.com/predict'),
      );
      
      request.files.add(await http.MultipartFile.fromPath('image', _image!.path));
      
      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      
      if (response.statusCode != 200) {
        setState(() {
          _result = 'Server error: ${response.statusCode} - $responseData';
        });
        return;
      }
      
      var jsonData = json.decode(responseData);

      setState(() {
        if (jsonData['predictions'].isNotEmpty) {
          _result = 'Detected: ${jsonData['predictions'].map((p) => '${p['class']} (${(p['confidence'] * 100).toStringAsFixed(1)}%)').join(', ')}';
        } else {
          _result = 'No face palsy detected';
        }
      });
    } catch (e) {
      setState(() {
        _result = 'Error: $e';
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Face Palsy Detector')),
      body: Column(
        children: [
          Expanded(
            child: _image != null
                ? Image.file(_image!, fit: BoxFit.contain)
                : Center(child: Text('No image selected')),
          ),
          if (_result.isNotEmpty)
            Padding(
              padding: EdgeInsets.all(16),
              child: Text(_result, style: TextStyle(fontSize: 16)),
            ),
          if (_loading)
            CircularProgressIndicator(),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                onPressed: _captureImage,
                child: Text('Capture'),
              ),
              ElevatedButton(
                onPressed: _image != null && !_loading ? _analyzeImage : null,
                child: Text('Analyze'),
              ),
            ],
          ),
          SizedBox(height: 20),
        ],
      ),
    );
  }
}