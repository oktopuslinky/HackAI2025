import { useState } from "react"
import FileUpload from "@/components/FileUpload"

export default function UploadPage() {
  const [uploadedFile, setUploadedFile] = useState(null)

  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center px-6 py-20 overflow-x-hidden space-y-20">
      <div className="text-center space-y-4 animate-fade-in">
        <h1 className="text-6xl md:text-7xl font-extrabold text-center tracking-tight leading-tight">
          Upload Your Annual Report
        </h1>
        <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto">
          Drag and drop your PDF file or browse to upload. We support files up to 50MB.
        </p>
      </div>

      <FileUpload onUpload={setUploadedFile} />

      {/* Extra content so page feels full */}
      <div className="text-center max-w-3xl space-y-6 animate-fade-in-slow">
        <h2 className="text-4xl font-bold">Why upload?</h2>
        <p className="text-gray-400 text-lg">
          We'll help you extract insights, generate executive summaries, and answer deep questions about your annual report.
        </p>
        <p className="text-gray-500">
          Secure, private, and optimized for enterprise-grade document analysis.
        </p>
      </div>
    </div>
  )
}
