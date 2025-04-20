"use client"

import { useState } from "react"
import UploadPage from "./pages/UploadPage"
import PdfViewer from "./pages/PdfViewer"
import "./App.css"

export default function App() {
  const [uploadedFile, setUploadedFile] = useState(null)
  const [currentView, setCurrentView] = useState("upload") // "upload" | "view"

  const handleFileUpload = (file) => {
    setUploadedFile(file)
    setCurrentView("view")
  }

  const handleBackToUpload = () => {
    setCurrentView("upload")
  }

  return (
    <>
      {currentView === "upload" && <UploadPage onFileUpload={handleFileUpload} />}
      {currentView === "view" && <PdfViewer file={uploadedFile} onBack={handleBackToUpload} />}
    </>
  )
}
