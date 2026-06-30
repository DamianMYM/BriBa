import Foundation
import Vision
import AppKit

struct OCRLine: Codable {
    let text: String
    let confidence: Float
}

struct OCRFrame: Codable {
    let file: String
    let lines: [OCRLine]
}

func recognizeText(at imageURL: URL) throws -> OCRFrame {
    guard let image = NSImage(contentsOf: imageURL) else {
        throw NSError(
            domain: "VisionOCR",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Failed to load image: \(imageURL.path)"]
        )
    }

    guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        throw NSError(
            domain: "VisionOCR",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Failed to create CGImage: \(imageURL.path)"]
        )
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = ["en-US", "zh-Hans"]
    request.minimumTextHeight = 0.01

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try handler.perform([request])

    let observations = request.results ?? []
    let lines = observations.compactMap { observation -> OCRLine? in
        guard let candidate = observation.topCandidates(1).first else {
            return nil
        }
        let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            return nil
        }
        return OCRLine(text: text, confidence: candidate.confidence)
    }

    return OCRFrame(file: imageURL.lastPathComponent, lines: lines)
}

let arguments = CommandLine.arguments
guard arguments.count >= 2 else {
    fputs("Usage: vision_ocr.swift <image_dir>\n", stderr)
    exit(1)
}

let imageDir = URL(fileURLWithPath: arguments[1], isDirectory: true)
let fileManager = FileManager.default
let imageURLs = try fileManager.contentsOfDirectory(at: imageDir, includingPropertiesForKeys: nil)
    .filter { url in
        let ext = url.pathExtension.lowercased()
        return ["png", "jpg", "jpeg"].contains(ext)
    }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

var frames: [OCRFrame] = []
frames.reserveCapacity(imageURLs.count)

for imageURL in imageURLs {
    do {
        frames.append(try recognizeText(at: imageURL))
    } catch {
        frames.append(OCRFrame(file: imageURL.lastPathComponent, lines: []))
    }
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(frames)
FileHandle.standardOutput.write(data)
