import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { UploadCloud, Trash2 } from "lucide-react"
import { useNavigate } from "react-router-dom"

import axios from "axios"

export default function FileUpload({ onUpload }) {
  const fileInputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      onUpload(selectedFile)
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current.click()
  }

  const handleRemoveFile = () => {
    setFile(null)
    setSuccess(false)
    onUpload(null)
  }

  async function handleContinue() {
    if (!file) {
      console.error("No file available to upload.")
      return
    }
  
    setLoading(true)
  
    try {
      // Always create FormData first
      const formData = new FormData()
      formData.append('file', file)
  
      const response = await axios.post('http://127.0.0.1:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 10000, // optional timeout safeguard
      })
  
      console.log(response.data)
      setSuccess(true)
  
      setTimeout(() => {
        navigate("/chatbot")
      }, 1000)
  
    } catch (error) {
      console.error("Upload failed:", error)
    } finally {
      setLoading(false)
    }
  }
  
  

  return (
    <div className="w-full flex flex-col items-center space-y-8">
      <div className="w-full max-w-2xl p-10 bg-[#111111] border border-gray-700 rounded-2xl shadow-md hover:shadow-lg transition-all duration-300 flex flex-col items-center space-y-6 animate-float-up">
        {!file ? (
          <>
            <UploadCloud className="h-16 w-16 text-gray-400 animate-pulse" />
            <h2 className="text-2xl font-semibold text-center text-white">
              Drop your PDF here
            </h2>
            <p className="text-gray-400 text-sm text-center">
              Drag and drop or browse to upload. Max file size: 50MB
            </p>
            <Input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              className="hidden"
              onChange={handleFileChange}
            />
            <Button
              variant="outline"
              className="border-gray-600 text-white hover:bg-gray-800 transition-all"
              onClick={handleUploadClick}
            >
              Select Document
            </Button>
          </>
        ) : (
          <>
            <div className="flex flex-col items-center space-y-2">
              <p className="text-lg font-semibold">{file.name}</p>
              <p className="text-gray-400 text-sm">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
            <Button
              variant="destructive"
              onClick={handleRemoveFile}
              className="flex items-center gap-2"
            >
              <Trash2 className="h-4 w-4" />
              Remove File
            </Button>
          </>
        )}
      </div>

      {file && !success && (
        <Button
          variant="default"
          className={`mt-8 bg-gradient-to-r from-purple-500 to-indigo-500 text-white px-8 py-3 rounded-xl text-lg font-semibold hover:opacity-90 transition-all ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
          onClick={handleContinue}
          disabled={loading}
        >
          {loading ? "Processing..." : "Continue"}
        </Button>
      )}

      {success && (
        <p className="text-green-400 mt-8 text-lg font-semibold animate-fade-in">
          Text extraction complete!
        </p>
      )}
    </div>
  )
}
