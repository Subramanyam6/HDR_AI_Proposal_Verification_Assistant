import { useEffect, useState } from 'react'
import { useAppStore } from './store/useAppStore'
import { Button } from './components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card'
import ReactMarkdown from 'react-markdown'
import './App.css'

function App() {
  const {
    inputText,
    pdfFile,
    results,
    isLoading,
    error,
    setInputText,
    setPdfFile,
    setSelectedSample,
    runVerification,
    clearAll,
  } = useAppStore()

  const [samples, setSamples] = useState<Record<string, string>>({})

  // Load samples on mount
  useEffect(() => {
    fetch('/api/samples')
      .then(res => res.json())
      .then(data => setSamples(data))
      .catch(err => console.error('Failed to load samples:', err))
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setPdfFile(file)
    }
  }

  const handleSampleChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const sampleKey = e.target.value
    if (sampleKey && sampleKey !== 'Select a text sample...') {
      await setSelectedSample(sampleKey)
    }
  }

  const renderModelResult = (value: boolean) => {
    return (
      <span className={value ? "text-red-600 font-semibold" : "text-green-600 font-semibold"}>
        {value ? "FAIL" : "PASS"}
      </span>
    )
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">HDR Proposal Verification Assistant</h1>
          <p className="text-muted-foreground">Upload a proposal PDF or paste text to verify compliance.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Panel: Input */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>INPUT</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* PDF Upload */}
                <div>
                  <label className="block text-sm font-semibold mb-2">UPLOAD A PROPOSAL PDF</label>
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    className="block w-full text-sm file:mr-4 file:py-2 file:px-4 file:border-2 file:border-black file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
                  />
                </div>

                <div className="text-center text-sm font-semibold">OR</div>

                {/* Text Input */}
                <div>
                  <label className="block text-sm font-semibold mb-2">COPY-PASTE THE PROPOSAL TEXT</label>
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Paste proposal text here..."
                    className="w-full h-64 p-3 border-2 border-black font-mono text-sm"
                  />
                  <select
                    onChange={handleSampleChange}
                    className="mt-2 w-full p-2 border-2 border-black"
                  >
                    <option>Select a text sample...</option>
                    {Object.keys(samples).map(key => (
                      <option key={key} value={key}>{key}</option>
                    ))}
                  </select>
                </div>

                {/* Buttons */}
                <div className="flex gap-4">
                  <Button
                    onClick={runVerification}
                    disabled={isLoading || (!inputText && !pdfFile)}
                    className="flex-1"
                    size="lg"
                  >
                    {isLoading ? "Processing..." : "Run Verification"}
                  </Button>
                  <Button
                    onClick={clearAll}
                    variant="destructive"
                    size="lg"
                  >
                    Clear
                  </Button>
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
              <CardHeader>
                <CardTitle>RESULTS</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {results && (
                  <>
                    {/* ML Results Table */}
                    <div>
                      <h3 className="font-semibold mb-3">Machine Learning-Based Checks</h3>
                      <table className="w-full border-2 border-black">
                        <thead>
                          <tr className="bg-muted">
                            <th className="border-2 border-black p-2 text-left">Check</th>
                            <th className="border-2 border-black p-2">Transformer</th>
                            <th className="border-2 border-black p-2">TF-IDF</th>
                            <th className="border-2 border-black p-2">Naive Bayes</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.keys(results.ml_results.transformer).map(key => (
                            <tr key={key}>
                              <td className="border-2 border-black p-2 font-medium">{key}</td>
                              <td className="border-2 border-black p-2 text-center">
                                {renderModelResult(results.ml_results.transformer[key as keyof typeof results.ml_results.transformer])}
                              </td>
                              <td className="border-2 border-black p-2 text-center">
                                {renderModelResult(results.ml_results.tfidf[key as keyof typeof results.ml_results.tfidf])}
                              </td>
                              <td className="border-2 border-black p-2 text-center">
                                {renderModelResult(results.ml_results.naive_bayes[key as keyof typeof results.ml_results.naive_bayes])}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Rule-Based Results */}
                    <div>
                      <h3 className="font-semibold mb-3">Rule-Based Checks</h3>
                      <table className="w-full border-2 border-black">
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
                              {renderModelResult(results.rule_results.date_inconsistency)}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>

                    {/* AI Suggestions */}
                    <div>
                      <h3 className="font-semibold mb-3">AI-Powered Fix Suggestions</h3>
                      <div className="p-4 border-2 border-black bg-muted">
                        <ReactMarkdown className="prose prose-sm max-w-none">
                          {results.suggestions}
                        </ReactMarkdown>
                      </div>
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

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-muted-foreground">
          <p>Powered by DistilBERT, TF-IDF + Logistic Regression, and GPT-4o</p>
        </div>
      </div>
    </div>
  )
}

export default App
