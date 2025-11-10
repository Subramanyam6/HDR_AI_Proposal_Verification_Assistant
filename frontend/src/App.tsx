import { useEffect, useState, useRef } from 'react'
import { useAppStore } from './store/useAppStore'
import { Button } from './components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card'
import { RotateCw, CloudUpload, X, Info, FileText } from 'lucide-react'
import { initializeIcons } from '@fluentui/react'
import FuzzyText from './components/FuzzyText'
import LetterGlitch from './components/LetterGlitch'
import DecryptedText from './components/DecryptedText'
import LicenseModal from './components/LicenseModal'
import AboutModal from './components/AboutModal'
import DocumentationSidebar from './components/DocumentationSidebar'
import CustomSelect from './components/CustomSelect'
import GlitchText from './components/GlitchText'
import DotGrid from './components/DotGrid'

const SAMPLE_DESCRIPTIONS: Record<string, string> = {
  'Clean proposal':
    'Baseline reference: Executive Summary, Work Approach, Team Availability, and the signature block all match—use this when you want to show a document with zero flags.',
  'Crosswalk error':
    'This sample keeps the Executive Summary crosswalk blurb on R4 but changes the Work Approach paragraph to cite R2, so the requirement ID is mismatched between Executive Summary and Work Approach.',
  'Banned phrases':
    'Identical to the clean file except the Work Approach paragraph now promises “delivery with absolute assurance,” which is one of the banned phrases.',
  'Name inconsistency':
    'Executive Summary and Point of Contact still say “Kevin Vazquez,” but the Team Availability table shortens it to “K. Vazquez,” creating inconsistent names across sections.',
  'Date inconsistency (rule)':
    'Executive Summary promises submission on 2025-12-12 while the signature sentence in the Point of Contact block says “signed and sealed on 2026-01-05,” so the signing date trails the promised date.'
}

initializeIcons()

function App() {
  const {
    inputText,
    pdfFile,
    selectedSample,
    selectedPdfSample,
    samples,
    results,
    isLoading,
    isSampleLoading,
    error,
    setInputText,
    setPdfFile,
    setSelectedSample,
    setSelectedPdfSample,
    loadSamples,
    runVerification,
    regenerateSuggestions,
    clearAll,
  } = useAppStore()

  const [ruleResultsLoading, setRuleResultsLoading] = useState(false)
  const [mlResultsLoading, setMlResultsLoading] = useState(false)
  const [mlResultsReadyTime, setMlResultsReadyTime] = useState<number | null>(null)
  const [suggestionsLoading, setSuggestionsLoading] = useState(false)
  const [suggestionsStartTime, setSuggestionsStartTime] = useState<number | null>(null)
  const [aiButtonClicked, setAiButtonClicked] = useState(false)
  const mlTimerRef = useRef<number | null>(null)
  const aiTimerRef = useRef<number | null>(null)

  // Load samples on mount
  useEffect(() => {
    loadSamples()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Handle rule-based results - show glitch immediately, load in real-time
  useEffect(() => {
    if (isLoading) {
      setRuleResultsLoading(true)
    } else if (results) {
      // Rule-based results arrive - show immediately (no wait)
      setRuleResultsLoading(false)
    } else {
      setRuleResultsLoading(false)
    }
  }, [isLoading, results])

  // Handle ML results loading state - glitch for 3 seconds AFTER results arrive
  useEffect(() => {
    if (isLoading) {
      // Clear timers if starting new verification
      if (mlTimerRef.current) {
        clearTimeout(mlTimerRef.current)
        mlTimerRef.current = null
      }
      if (aiTimerRef.current) {
        clearTimeout(aiTimerRef.current)
        aiTimerRef.current = null
      }
      setMlResultsLoading(true)
      setMlResultsReadyTime(null)
      setSuggestionsStartTime(null)
      setAiButtonClicked(false)
    } else if (!results) {
      // Clear timers if no results
      if (mlTimerRef.current) {
        clearTimeout(mlTimerRef.current)
        mlTimerRef.current = null
      }
      if (aiTimerRef.current) {
        clearTimeout(aiTimerRef.current)
        aiTimerRef.current = null
      }
      setMlResultsLoading(false)
      setMlResultsReadyTime(null)
      setSuggestionsStartTime(null)
      setAiButtonClicked(false)
    }
  }, [isLoading, results])

  // Start ML timer when results arrive (only once)
  useEffect(() => {
    // Only start timer if conditions are met and timer isn't already running
    if (results && !isLoading && mlResultsReadyTime === null && !mlTimerRef.current) {
      console.log('Starting ML timer')
      const readyTime = Date.now()
      setMlResultsReadyTime(readyTime)
      
      mlTimerRef.current = setTimeout(() => {
        console.log('ML timer fired - showing results')
        setMlResultsLoading(false)
        mlTimerRef.current = null
      }, 3000)
    }
  }, [results, isLoading, mlResultsReadyTime])

  // Start AI timer when button is clicked
  useEffect(() => {
    // Only start timer if button clicked, results exist, ML done, timer not running, and startTime not set
    if (aiButtonClicked && results && !mlResultsLoading && !aiTimerRef.current && suggestionsStartTime === null) {
      const startTime = Date.now()
      console.log('Starting AI suggestions timer')
      setSuggestionsStartTime(startTime)
      setSuggestionsLoading(true)
      
      aiTimerRef.current = setTimeout(() => {
        console.log('AI timer fired - showing suggestions')
        setSuggestionsLoading(false)
        aiTimerRef.current = null
      }, 5000)
    }
    
    // Cleanup only when resetting (not when suggestionsStartTime changes)
    return () => {
      if (!aiButtonClicked || !results || mlResultsLoading) {
        if (aiTimerRef.current) {
          clearTimeout(aiTimerRef.current)
          aiTimerRef.current = null
        }
      }
    }
  }, [aiButtonClicked, results, mlResultsLoading]) // Removed suggestionsStartTime to prevent cleanup loop

  const [isDragging, setIsDragging] = useState(false)
  const [licenseModalOpen, setLicenseModalOpen] = useState(false)
  const [aboutModalOpen, setAboutModalOpen] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setPdfFile(file)
    } else {
      setPdfFile(null)
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type === 'application/pdf') {
      setPdfFile(file)
    }
  }


  const handleClearText = () => {
    setInputText('')
    setSelectedSample(null)
  }

  const handleClearPdf = () => {
    setPdfFile(null)
    setSelectedPdfSample(null)
  }

  const [fuzzyTextValues, setFuzzyTextValues] = useState<Record<string, boolean>>({})

  // Randomly alternate fuzzy text between PASS/FAIL every second
  useEffect(() => {
    if (ruleResultsLoading || mlResultsLoading) {
      const interval = setInterval(() => {
        setFuzzyTextValues(_prev => {
          const newValues: Record<string, boolean> = {}
          // Generate random PASS/FAIL for each key
          const keys = ['date-rule', 'crosswalk-transformer', 'crosswalk-tfidf', 'crosswalk-nb', 
                       'banned-transformer', 'banned-tfidf', 'banned-nb',
                       'name-transformer', 'name-tfidf', 'name-nb']
          keys.forEach(key => {
            newValues[key] = Math.random() > 0.5
          })
          return newValues
        })
      }, 1000)
      
      return () => clearInterval(interval)
    } else {
      // Clear fuzzy values when not loading
      setFuzzyTextValues({})
    }
  }, [ruleResultsLoading, mlResultsLoading])

  const renderModelResult = (value: boolean, isLoading: boolean, key?: string) => {
    if (isLoading && key) {
      // Show glitching PASS/FAIL that randomly alternates every second
      const showFail = fuzzyTextValues[key] ?? (Math.random() > 0.5)
      return (
        <FuzzyText
          baseIntensity={0.15}
          hoverIntensity={0.15}
          enableHover={false}
          fontSize="1.40625rem"
          fontWeight={600}
          color="#000000"
        >
          {showFail ? "FAIL" : "PASS"}
        </FuzzyText>
      )
    }
    return (
      <span className={`${value ? "text-red-600" : "text-green-600"} font-semibold text-base`}>
        {value ? "FAIL" : "PASS"}
      </span>
    )
  }

  return (
    <div className="min-h-screen bg-background flex flex-col relative">
      <div className="fixed inset-0 -z-10">
        <LetterGlitch glitchSpeed={50} centerVignette={false} outerVignette={true} smooth={true} />
      </div>

      {/* Header */}
      <header className="border-b-4 border-black bg-transparent relative z-50 overflow-x-hidden overflow-y-visible">
        <DotGrid 
          dotSize={4.6}
          gap={4}
          baseColor="#D1D5DB"
          activeColor="#3B82F6"
          proximity={100}
          className="absolute inset-0"
          style={{ opacity: 0.575 }}
        />
        <div className="w-full pl-[calc(2rem*0.95)] pr-8 py-[calc(1.5rem*0.50*0.70*0.85)] relative overflow-x-hidden overflow-y-visible z-10">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="mb-2 block flex items-center gap-2">
                <GlitchText enableOnHover={true}>HDR</GlitchText>
                <GlitchText enableOnHover={true}>Proposal</GlitchText>
                <GlitchText enableOnHover={true}>Verification</GlitchText>
                <GlitchText enableOnHover={true}>Assistant</GlitchText>
              </h1>
              <p className="text-foreground/90 text-lg block m-0 p-0 flex items-center gap-2 font-bold">
                <DecryptedText
                  text="Machine-learning-based compliance checks"
                  className="text-foreground/90 font-bold"
                  animateOn="view"
                  sequential={true}
                  revealDirection="start"
                  speed={75}
                />
                <span className="header-pipeline-text">|</span>
                <DecryptedText
                  text="LLM-assisted fix suggestions"
                  className="text-foreground/90 font-bold"
                  animateOn="view"
                  sequential={true}
                  revealDirection="start"
                  speed={75}
                />
              </p>
            </div>
            <div className="flex items-center gap-4 relative overflow-visible">
              <a
                href="https://github.com/Subramanyam6/HDR_AI_Proposal_Verification_Assistant"
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground hover:text-accent transition-all duration-200 group relative z-10"
                aria-label="GitHub Repository"
              >
                <svg className="w-12 h-12 group-hover:scale-125 transition-transform duration-200" viewBox="0 0 24 24" fill="currentColor">
                  <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
                </svg>
              </a>
              <div className="header-pipeline-inline"></div>
              <a
                href="https://huggingface.co/spaces/Subramanyam6/HDR_AI_Proposal_Verification_Assistant_V2"
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground transition-all duration-200 group flex items-center justify-center w-12 h-12 relative z-10"
                aria-label="HuggingFace Space"
              >
                <span className="text-4xl leading-none group-hover:scale-125 transition-transform duration-200">🤗</span>
              </a>
              <div className="header-pipeline-inline"></div>
              <button
                onClick={() => setAboutModalOpen(true)}
                className={`transition-all duration-200 group relative z-10 flex items-center justify-center w-12 h-12 ${
                  aboutModalOpen ? 'text-primary' : 'text-foreground hover:text-[#3B82F6]'
                }`}
                aria-label="About"
              >
                <Info className="w-12 h-12 group-hover:scale-125 transition-transform duration-200" strokeWidth={2} />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Sidebar */}
        <DocumentationSidebar />

        {/* Main Content */}
        <main className="flex-1 p-8 overflow-y-auto">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Panel: Input */}
          <div className="space-y-6">
            <Card>
              <CardHeader className="bg-secondary border-b-4 border-black py-[calc(0.5rem*1.81)] min-h-0 flex items-center justify-center">
                <CardTitle className="text-center text-black font-bold text-base leading-tight">INPUT</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* PDF Upload */}
                <div className="mt-4">
                  <label className="block text-sm font-semibold mb-2">UPLOAD A PROPOSAL PDF</label>
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`relative border-2 border-black bg-background transition-colors shadow-brutal p-6 text-center ${pdfFile ? 'cursor-default' : 'cursor-pointer pdf-upload-hover'} ${isDragging ? 'pdf-upload-dragging' : ''} ${inputText ? 'opacity-50 cursor-not-allowed pointer-events-none' : ''}`}
                    onClick={() => !inputText && !pdfFile && document.getElementById('pdf-input')?.click()}
                  >
                    <input
                      id="pdf-input"
                      type="file"
                      accept=".pdf"
                      onChange={handleFileChange}
                      disabled={!!inputText}
                      className="hidden"
                      key={`file-input-${pdfFile?.name || selectedPdfSample || 'default'}`}
                    />
                    {pdfFile && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleClearPdf()
                        }}
                        className="absolute top-2 right-2 p-1 bg-destructive text-destructive-foreground hover:bg-red-600 border-2 border-black shadow-brutal z-50 transition-colors"
                        aria-label="Remove PDF"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}
                    {pdfFile ? (
                      <div className="space-y-3">
                        <FileText className="w-12 h-12 mx-auto text-foreground" style={{ strokeWidth: 2, strokeLinecap: 'square', strokeLinejoin: 'miter' }} />
                        <div>
                          <p className="text-sm font-semibold mb-1">
                            {pdfFile.name}
                          </p>
                          <p className="text-xs text-muted-foreground">PDF uploaded successfully</p>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <CloudUpload className="w-12 h-12 mx-auto text-foreground" style={{ strokeWidth: 2, strokeLinecap: 'square', strokeLinejoin: 'miter' }} />
                        <div>
                          <p className="text-sm font-semibold mb-1">
                            Drag and drop your PDF here
                          </p>
                          <p className="text-xs text-muted-foreground">or</p>
                          <p className="text-sm font-semibold mt-1">Click to browse</p>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="mt-4">
                    <CustomSelect
                      value={selectedPdfSample || ''}
                      onChange={(value) => setSelectedPdfSample(value)}
                      options={Object.keys(samples)}
                      placeholder="Select a PDF sample to upload..."
                      disabled={isSampleLoading || !!inputText}
                      optionDetails={SAMPLE_DESCRIPTIONS}
                    />
                  </div>
                </div>

                <div className="text-center text-sm font-semibold">OR</div>

                {/* Text Input */}
                <div className={pdfFile ? 'opacity-50 pointer-events-none' : ''}>
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-sm font-semibold">COPY-PASTE THE PROPOSAL TEXT</label>
                    {isSampleLoading && (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <RotateCw className="w-4 h-4 animate-spin" />
                        <span>Loading sample...</span>
                      </div>
                    )}
                  </div>
                  <div className="relative">
                    <textarea
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      placeholder="Paste proposal text here..."
                      disabled={isSampleLoading || !!pdfFile}
                      className="w-full h-64 p-3 pr-10 border-2 border-black font-mono text-sm bg-background disabled:opacity-50 disabled:cursor-not-allowed shadow-brutal resize-y"
                      style={{ minHeight: '16rem' }}
                    />
                    {inputText && (
                      <button
                        onClick={handleClearText}
                        className="absolute top-2 right-2 p-1 bg-destructive text-destructive-foreground hover:bg-red-600 border-2 border-black shadow-brutal transition-colors"
                        aria-label="Clear text"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                  <div className="mt-2">
                    <CustomSelect
                      value={selectedSample || ''}
                      onChange={(value) => setSelectedSample(value)}
                      options={Object.keys(samples)}
                      placeholder="Select a text sample to upload..."
                      disabled={isSampleLoading || !!pdfFile}
                      optionDetails={SAMPLE_DESCRIPTIONS}
                    />
                  </div>
                </div>

                {/* Buttons */}
                <div className="flex gap-4">
                  <div className="w-full">
                    <Button
                      onClick={runVerification}
                      disabled={isLoading || (!inputText && !pdfFile)}
                      className="w-full font-semibold border-2 border-black bg-[#10b981] hover:bg-[#059669] text-white disabled:bg-gray-400 disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:bg-gray-400"
                      size="lg"
                    >
                      {isLoading ? "Processing..." : "Click to Run Verification"}
                    </Button>
                  </div>
                </div>

                {error && (
                  <div className="p-3 border-2 border-red-600 bg-red-50 text-red-900 text-sm">
                    ⚠ {error}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Panel: Results */}
          <div className="space-y-6">
            <Card>
              <CardHeader className="bg-secondary border-b-4 border-black py-2 min-h-0 relative">
                <div className="flex items-center justify-between w-full">
                  <CardTitle className="flex-1 text-center text-black font-bold text-base leading-tight">RESULTS</CardTitle>
                  <div className="flex-shrink-0" style={{ transform: 'translateY(-9%)' }}>
                    <Button
                      onClick={clearAll}
                      disabled={!results}
                      size="sm"
                      className="disabled:bg-gray-400 disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:bg-gray-400 hover:translate-x-0 hover:translate-y-0"
                    >
                      Clear
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Show layout immediately when loading or results exist */}
                {(isLoading || results) && (
                  <>
                    {/* Rule-Based Results - Load in real-time with glitch */}
                    <div>
                      <div className="flex items-center gap-3 mb-4 mt-2">
                        <h3 className="font-semibold">Rule-Based Checks</h3>
                        {ruleResultsLoading && (
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <RotateCw className="w-4 h-4 animate-spin" />
                            <span>Loading</span>
                          </div>
                        )}
                      </div>
                      <table className="w-full border-2 border-black shadow-brutal">
                        <thead>
                          <tr className="bg-muted">
                            <th className="border-2 border-black p-2 text-left">Check</th>
                            <th className="border-2 border-black p-2">Python & Regex</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td className="border-2 border-black p-2 font-medium">Date Inconsistency</td>
                            <td className="border-2 border-black p-2 text-center">
                              {ruleResultsLoading ? renderModelResult(false, true, 'date-rule') : (results ? renderModelResult(results.rule_results.date_inconsistency, false, 'date-rule') : '-')}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>

                    {/* ML Results Table */}
                    <div>
                      <div className="flex items-center gap-3 mb-3">
                        <h3 className="font-semibold">Machine Learning-Based Checks</h3>
                        {mlResultsLoading && !ruleResultsLoading && (
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <RotateCw className="w-4 h-4 animate-spin" />
                            <span>Loading</span>
                          </div>
                        )}
                      </div>
                      <table className="w-full border-2 border-black shadow-brutal">
                        <thead>
                          <tr className="bg-muted">
                            <th className="border-2 border-black p-2 text-left">Check</th>
                            <th className="border-2 border-black p-2">
                              DistilBERT
                              <span className="ml-1 text-yellow-500 cursor-default relative group">
                                <span className="text-xl">*</span>
                                <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-black text-white text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 border-2 border-black shadow-brutal">
                                  Most Accurate
                                </span>
                              </span>
                            </th>
                            <th className="border-2 border-black p-2">TF-IDF</th>
                            <th className="border-2 border-black p-2">Naive Bayes</th>
                          </tr>
                        </thead>
                        <tbody>
                          {mlResultsLoading && !ruleResultsLoading ? (
                            // Show glitching PASS/FAIL while ML is loading (after rule-based finishes)
                            <>
                              <tr>
                                <td className="border-2 border-black p-2 font-medium">Crosswalk Error</td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(false, true, 'crosswalk-transformer')}
                                </td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(false, true, 'crosswalk-tfidf')}
                                </td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(false, true, 'crosswalk-nb')}
                                </td>
                              </tr>
                              <tr>
                                <td className="border-2 border-black p-2 font-medium">Banned Phrases</td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(false, true, 'banned-transformer')}
                                </td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(false, true, 'banned-tfidf')}
                                </td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(false, true, 'banned-nb')}
                                </td>
                              </tr>
                              <tr>
                                <td className="border-2 border-black p-2 font-medium">Name Inconsistency</td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(false, true, 'name-transformer')}
                                </td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(false, true, 'name-tfidf')}
                                </td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(false, true, 'name-nb')}
                                </td>
                              </tr>
                            </>
                          ) : results ? (
                            // Show actual results when ML loading is done
                            Object.keys(results.ml_results.transformer).map(key => (
                              <tr key={key}>
                                <td className="border-2 border-black p-2 font-medium">{key}</td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(results.ml_results.transformer[key as keyof typeof results.ml_results.transformer], false, `${key}-transformer`)}
                                </td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(results.ml_results.tfidf[key as keyof typeof results.ml_results.tfidf], false, `${key}-tfidf`)}
                                </td>
                                <td className="border-2 border-black p-2 text-center">
                                  {renderModelResult(results.ml_results.naive_bayes[key as keyof typeof results.ml_results.naive_bayes], false, `${key}-nb`)}
                                </td>
                              </tr>
                            ))
                          ) : (
                            // Show dashes before any loading starts
                            <>
                              <tr>
                                <td className="border-2 border-black p-2 font-medium">Crosswalk Error</td>
                                <td className="border-2 border-black p-2 text-center">-</td>
                                <td className="border-2 border-black p-2 text-center">-</td>
                                <td className="border-2 border-black p-2 text-center">-</td>
                              </tr>
                              <tr>
                                <td className="border-2 border-black p-2 font-medium">Banned Phrases</td>
                                <td className="border-2 border-black p-2 text-center">-</td>
                                <td className="border-2 border-black p-2 text-center">-</td>
                                <td className="border-2 border-black p-2 text-center">-</td>
                              </tr>
                              <tr>
                                <td className="border-2 border-black p-2 font-medium">Name Inconsistency</td>
                                <td className="border-2 border-black p-2 text-center">-</td>
                                <td className="border-2 border-black p-2 text-center">-</td>
                                <td className="border-2 border-black p-2 text-center">-</td>
                              </tr>
                            </>
                          )}
                        </tbody>
                      </table>
                    </div>

                    {/* AI Suggestions */}
                    <div>
                      <div className="flex items-center gap-3 mb-3">
                        <div className="flex items-center gap-2">
                          <Button
                            onClick={async () => {
                              if (!mlResultsLoading && results) {
                                // Regenerate suggestions only (not full verification)
                                setAiButtonClicked(true)
                                setSuggestionsStartTime(null)
                                setSuggestionsLoading(true)
                                if (aiTimerRef.current) {
                                  clearTimeout(aiTimerRef.current)
                                  aiTimerRef.current = null
                                }
                                
                                // Start timer for minimum 5 seconds
                                const startTime = Date.now()
                                setSuggestionsStartTime(startTime)
                                
                                // Fetch new suggestions
                                const suggestionsPromise = regenerateSuggestions()
                                
                                // Wait for both API call and minimum 5 seconds
                                const [_, __] = await Promise.all([
                                  suggestionsPromise,
                                  new Promise(resolve => setTimeout(resolve, 5000))
                                ])
                                
                                // Both completed, stop loading
                                if (aiTimerRef.current) {
                                  clearTimeout(aiTimerRef.current)
                                  aiTimerRef.current = null
                                }
                                setSuggestionsLoading(false)
                              }
                            }}
                            disabled={mlResultsLoading || !results}
                            className="font-semibold border-2 border-black bg-[#10b981] hover:bg-[#059669] text-white disabled:bg-gray-400 disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:bg-gray-400"
                          >
                            Click for AI-Powered Fix Suggestions
                          </Button>
                          <div className="flex items-center gap-2 ml-1">
                            <span className="text-sm text-muted-foreground self-center">—</span>
                            <div className="flex flex-col text-sm text-muted-foreground leading-tight">
                              <span>DistilBERT</span>
                              <span>Transformer</span>
                            </div>
                          </div>
                        </div>
                        {(suggestionsLoading || (aiButtonClicked && !results?.suggestions)) && !mlResultsLoading && (
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <RotateCw className="w-4 h-4 animate-spin" />
                            <span>Loading</span>
                          </div>
                        )}
                      </div>
                      {aiButtonClicked ? (
                        <div className="p-4 border-2 border-black bg-background prose prose-sm max-w-none relative min-h-[200px] overflow-hidden shadow-brutal">
                          {suggestionsLoading ? (
                            <div className="absolute inset-0 w-full h-full">
                              <LetterGlitch
                                glitchSpeed={50}
                                centerVignette={false}
                                outerVignette={false}
                                smooth={true}
                                glitchColors={['#FF6B35', '#FFF44F', '#FFFFFF', '#000000']}
                              />
                            </div>
                          ) : results?.suggestions ? (
                            <div 
                              dangerouslySetInnerHTML={{ __html: results.suggestions }}
                            />
                          ) : (
                            <div className="flex items-center justify-center h-full min-h-[200px]">
                              <p className="text-sm text-muted-foreground">Generating suggestions...</p>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="p-4 border-2 border-black bg-background prose prose-sm max-w-none relative min-h-[200px] flex items-center justify-center shadow-brutal">
                          <p className="text-sm text-muted-foreground">
                            {mlResultsLoading || !results ? "Waiting for the label checks to complete..." : "Click the button above to generate AI-powered fix suggestions"}
                          </p>
                        </div>
                      )}
                    </div>
                  </>
                )}

                {!results && !isLoading && (
                  <div className="text-center text-muted-foreground py-12">
                    Run verification to see results here
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
        </div>
      </main>
      </div>

      {/* Footer */}
      <footer className="border-t-4 border-black bg-secondary mt-8 relative z-50 overflow-x-hidden overflow-y-visible">
        <div className="w-full py-6 h-full">
          <div className="relative h-full overflow-x-hidden overflow-y-visible">
            <div className="footer-scroll whitespace-nowrap h-full flex items-center">
              {/* Duplicate content for seamless loop */}
              {[...Array(2)].map((_, loopIndex) => (
                <div key={loopIndex} className="flex items-center gap-1.5 px-8 h-full flex-shrink-0">
                  <span className="text-base font-bold flex-shrink-0">Built with ❤️ by Bala for HDR</span>
                  <svg 
                    className="w-10 h-10 flex-shrink-0" 
                    viewBox="0 0 24 24" 
                    fill="none"
                    style={{ verticalAlign: 'middle' }}
                  >
                    <path d="M12 2L14 8L10 8L12 2Z" fill="#D4AF37" stroke="#B8941F" strokeWidth="0.5" />
                    <line x1="10.5" y1="4.5" x2="13.5" y2="4.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="5.5" x2="13.5" y2="5.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="6.5" x2="13.5" y2="6.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <circle cx="12" cy="5.5" r="1.5" fill="#DC2626" />
                    <line x1="12" y1="8" x2="12" y2="22" stroke="#D4AF37" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  <span className="text-base font-bold flex-shrink-0">Minimum Viable Product</span>
                  <svg 
                    className="w-10 h-10 flex-shrink-0" 
                    viewBox="0 0 24 24" 
                    fill="none"
                    style={{ verticalAlign: 'middle' }}
                  >
                    <path d="M12 2L14 8L10 8L12 2Z" fill="#D4AF37" stroke="#B8941F" strokeWidth="0.5" />
                    <line x1="10.5" y1="4.5" x2="13.5" y2="4.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="5.5" x2="13.5" y2="5.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="6.5" x2="13.5" y2="6.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <circle cx="12" cy="5.5" r="1.5" fill="#DC2626" />
                    <line x1="12" y1="8" x2="12" y2="22" stroke="#D4AF37" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  <a
                    href="https://github.com/Subramanyam6/HDR_AI_Proposal_Verification_Assistant"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-foreground hover:text-accent transition-all duration-200 group flex-shrink-0"
                    aria-label="GitHub Repository"
                  >
                    <svg className="w-9 h-9 group-hover:scale-125 transition-transform duration-200" viewBox="0 0 24 24" fill="currentColor">
                      <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
                    </svg>
                  </a>
                  <svg 
                    className="w-10 h-10 flex-shrink-0" 
                    viewBox="0 0 24 24" 
                    fill="none"
                    style={{ verticalAlign: 'middle' }}
                  >
                    <path d="M12 2L14 8L10 8L12 2Z" fill="#D4AF37" stroke="#B8941F" strokeWidth="0.5" />
                    <line x1="10.5" y1="4.5" x2="13.5" y2="4.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="5.5" x2="13.5" y2="5.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="6.5" x2="13.5" y2="6.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <circle cx="12" cy="5.5" r="1.5" fill="#DC2626" />
                    <line x1="12" y1="8" x2="12" y2="22" stroke="#D4AF37" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  <a
                    href="https://huggingface.co/spaces/Subramanyam6/HDR_AI_Proposal_Verification_Assistant_V2"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-foreground transition-all duration-200 group flex items-center justify-center w-9 h-9 flex-shrink-0"
                    aria-label="HuggingFace Space"
                  >
                    <span className="text-3xl leading-none group-hover:scale-125 transition-transform duration-200">🤗</span>
                  </a>
                  <svg 
                    className="w-10 h-10 flex-shrink-0" 
                    viewBox="0 0 24 24" 
                    fill="none"
                    style={{ verticalAlign: 'middle' }}
                  >
                    <path d="M12 2L14 8L10 8L12 2Z" fill="#D4AF37" stroke="#B8941F" strokeWidth="0.5" />
                    <line x1="10.5" y1="4.5" x2="13.5" y2="4.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="5.5" x2="13.5" y2="5.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="6.5" x2="13.5" y2="6.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <circle cx="12" cy="5.5" r="1.5" fill="#DC2626" />
                    <line x1="12" y1="8" x2="12" y2="22" stroke="#D4AF37" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  <span className="text-base font-bold flex-shrink-0">© 2025</span>
                  <svg 
                    className="w-10 h-10 flex-shrink-0" 
                    viewBox="0 0 24 24" 
                    fill="none"
                    style={{ verticalAlign: 'middle' }}
                  >
                    <path d="M12 2L14 8L10 8L12 2Z" fill="#D4AF37" stroke="#B8941F" strokeWidth="0.5" />
                    <line x1="10.5" y1="4.5" x2="13.5" y2="4.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="5.5" x2="13.5" y2="5.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="6.5" x2="13.5" y2="6.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <circle cx="12" cy="5.5" r="1.5" fill="#DC2626" />
                    <line x1="12" y1="8" x2="12" y2="22" stroke="#D4AF37" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  <span className="text-base font-bold flex-shrink-0">Bala Subramanyam Duggirala</span>
                  <svg 
                    className="w-10 h-10 flex-shrink-0" 
                    viewBox="0 0 24 24" 
                    fill="none"
                    style={{ verticalAlign: 'middle' }}
                  >
                    <path d="M12 2L14 8L10 8L12 2Z" fill="#D4AF37" stroke="#B8941F" strokeWidth="0.5" />
                    <line x1="10.5" y1="4.5" x2="13.5" y2="4.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="5.5" x2="13.5" y2="5.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="6.5" x2="13.5" y2="6.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <circle cx="12" cy="5.5" r="1.5" fill="#DC2626" />
                    <line x1="12" y1="8" x2="12" y2="22" stroke="#D4AF37" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  <span className="text-base font-bold flex-shrink-0">HDR Inc.</span>
                  <svg 
                    className="w-10 h-10 flex-shrink-0" 
                    viewBox="0 0 24 24" 
                    fill="none"
                    style={{ verticalAlign: 'middle' }}
                  >
                    <path d="M12 2L14 8L10 8L12 2Z" fill="#D4AF37" stroke="#B8941F" strokeWidth="0.5" />
                    <line x1="10.5" y1="4.5" x2="13.5" y2="4.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="5.5" x2="13.5" y2="5.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <line x1="10.5" y1="6.5" x2="13.5" y2="6.5" stroke="white" strokeWidth="1" strokeLinecap="round" />
                    <circle cx="12" cy="5.5" r="1.5" fill="#DC2626" />
                    <line x1="12" y1="8" x2="12" y2="22" stroke="#D4AF37" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  <button
                    onClick={() => setLicenseModalOpen(true)}
                    className="text-base font-semibold border-2 border-black px-6 py-3 bg-background hover:bg-accent hover:text-accent-foreground transition-colors shadow-brutal hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none flex-shrink-0"
                  >
                    Apache 2.0 License
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </footer>

      {/* License Modal */}
      <LicenseModal isOpen={licenseModalOpen} onClose={() => setLicenseModalOpen(false)} />
      <AboutModal isOpen={aboutModalOpen} onClose={() => setAboutModalOpen(false)} />
    </div>
  )
}

export default App
