import { useState } from 'react'
import { BrainCircuit, Zap, ShieldAlert, ChevronDown } from 'lucide-react'

type ModelStat = {
  name: string
  description: string
  metrics: {
    micro: number
    macro: number
  }
  role: string
  icon: React.ElementType
  legend: string
}

type LabelInfo = {
  title: string
  focus: string
  trigger: string
  section: string
}

const PROPOSAL_SECTIONS = [
  {
    title: 'Executive Summary',
    description: 'Opens the letter, states intent, and introduces the Project Manager by name.'
  },
  {
    title: 'Team Availability',
    description: 'Quick table showing who is staffed on the job and their percentage commitment.'
  },
  {
    title: 'Work Approach',
    description: 'Plain-language story of how we will deliver and where the R1–R5 requirement IDs live.'
  },
  {
    title: 'Schedule',
    description: 'Two sentences: expected submission date and confirmation of when it was signed.'
  },
  {
    title: 'Point of Contact',
    description: 'Contact block containing PM name, phone, email, and signature line.'
  }
]

const MODEL_STATS: ModelStat[] = [
  {
    name: 'DistilBERT',
    description: 'Uses advanced language understanding to read entire sentences and catch subtle errors that might be missed by simple word matching. Think of it as reading for meaning, not just keywords.',
    metrics: { micro: 0.956, macro: 0.953 },
    role: '',
    icon: BrainCircuit,
    legend: ''
  },
  {
    name: 'TF-IDF + Logistic Regression',
    description: 'Analyzes how often specific words and phrases appear in the text. It\'s particularly good at spotting exact phrases that shouldn\'t be there, like "guaranteed success" or "risk-free delivery".',
    metrics: { micro: 0.851, macro: 0.884 },
    role: '',
    icon: Zap,
    legend: ''
  },
  {
    name: 'Naive Bayes',
    description: 'A fast, simple model that acts as a backup check. When the other two models disagree, this one helps break the tie by looking at basic word patterns.',
    metrics: { micro: 0.861, macro: 0.861 },
    role: '',
    icon: ShieldAlert,
    legend: ''
  }
]

const LABEL_DETAILS: LabelInfo[] = [
  {
    title: 'Crosswalk Error',
    focus: 'We zoom in on the Work Approach paragraph—the part that spells out the delivery plan and cites requirement numbers (R1, R2, ...).',
    trigger: 'Flagged when the requirement ID in that paragraph does not match the agency’s crosswalk table.',
    section: 'Think “Does the story mention the same requirement ID the official checklist expects?”'
  },
  {
    title: 'Banned Phrases',
    focus: 'We read the exact same Work Approach paragraph but only look for phrases such as “risk-free delivery” or “guaranteed success.”',
    trigger: 'Any forbidden wording anywhere in that paragraph triggers the label—no helper sentences or cues.',
    section: 'It’s the same plan paragraph; we simply search within it for those phrases.'
  },
  {
    title: 'Name Inconsistency',
    focus: 'We compare every place the Project Manager appears: greeting, staffing table, and contact block.',
    trigger: 'If any instance deviates from the base spelling (e.g., Sarah Martinez → Sarh Martinez), we flag it.',
    section: 'In other words, every mention of the PM must stay identical from top to bottom.'
  }
]

const DATA_FACTS = [
  { label: 'Clean starting docs', value: '5,000 proposals that already pass every check.' },
  { label: 'Twin versions', value: 'Each clean doc gets three twins: one with a crosswalk swap, one with a banned phrase, one with a name typo.' },
  { label: 'Balanced splits', value: '20,000 total records divided across training, validation, and test just like production.' },
  { label: 'Length safety', value: 'We trim or rewrite so 95%+ of the records stay under 512 tokens (the model limit).' }
]

const DATE_PATTERNS = [
  'YYYY-MM-DD',
  'MM/DD/YYYY • DD/MM/YYYY',
  'Month DD, YYYY',
  'DD Month YYYY'
]

function AccordionCard({
  title,
  children,
  defaultOpen = false,
}: {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  return (
    <div className="border-2 border-black shadow-brutal">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full px-4 py-2 font-semibold text-left border-b-2 border-black transition-colors duration-200 flex items-center justify-between ${
          isOpen
            ? 'bg-primary text-primary-foreground'
            : 'bg-card text-card-foreground hover:bg-accent hover:text-accent-foreground'
        }`}
      >
        <span>{title}</span>
        <ChevronDown
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>
      {isOpen && <div className="p-4 bg-background space-y-3">{children}</div>}
    </div>
  )
}

export default function DocumentationSidebar() {
  return (
    <aside
      className="w-[320px] border-r-4 border-black bg-background/90 backdrop-blur flex flex-col sticky top-0 z-40"
      style={{ height: 'calc(100vh - 5.025rem)', maxHeight: 'calc(100vh - 5.025rem)' }}
    >
      <div className="overflow-y-auto flex-1 px-6 space-y-4 pt-6 pb-10">
        <AccordionCard title="LABELS" defaultOpen={false}>
          <div className="space-y-3">
            {LABEL_DETAILS.map(label => (
              <div key={label.title} className="bg-white border border-black/30 p-3 space-y-1 text-xs leading-relaxed">
                <p className="font-semibold text-sm">{label.title}</p>
                <p>{label.focus}</p>
                <p><span className="font-semibold">What triggers it?</span> {label.trigger}</p>
                <p className="text-muted-foreground">{label.section}</p>
              </div>
            ))}
          </div>
        </AccordionCard>

        <AccordionCard title="PROPOSAL STRUCTURE">
          <div className="space-y-2 text-xs leading-relaxed text-muted-foreground">
            {PROPOSAL_SECTIONS.map(section => (
              <div key={section.title} className="bg-white border border-black/20 p-2">
                <p className="font-semibold text-sm text-black">{section.title}</p>
                <p>{section.description}</p>
              </div>
            ))}
          </div>
        </AccordionCard>

        <AccordionCard title="ML-BASED CHECKS">
          <div className="space-y-3 text-xs leading-relaxed">
            <div className="bg-muted border border-black/30 p-2 mb-2">
              <p className="text-[10px] text-muted-foreground leading-relaxed">
                <span className="font-semibold">About F1-scores:</span> These numbers measure how accurate each model is. Higher values (closer to 1.0) mean the model is better at correctly identifying both problems and clean text. Micro F1 looks at overall accuracy, while Macro F1 averages performance across all error types.
              </p>
            </div>
            {MODEL_STATS.map(stat => (
              <div key={stat.name} className="bg-white border border-black/30 p-3 space-y-2 rounded-sm text-card-foreground break-words">
                <div className="flex items-center gap-2">
                  <stat.icon className="h-4 w-4 text-black flex-shrink-0" />
                  <span className="font-semibold text-sm">{stat.name}</span>
                </div>
                <p className="text-muted-foreground leading-relaxed">{stat.description}</p>
                <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono pt-1 border-t border-black/20">
                  <span className="text-card-foreground">
                    Micro F1-score: <span className="font-semibold">{stat.metrics.micro.toFixed(3)}</span>
                  </span>
                  <span className="text-card-foreground">
                    Macro F1-score: <span className="font-semibold">{stat.metrics.macro.toFixed(3)}</span>
                  </span>
                </div>
              </div>
            ))}
          </div>
        </AccordionCard>

        <AccordionCard title="REGEX-BASED CHECKS">
          <div className="space-y-2 text-xs text-muted-foreground leading-relaxed">
            <p>We match two specific sentences: the one that promises an “anticipated submission date” and the one that states “signed and sealed on…”. If the signing date comes later than the promised date, the system raises the Date Inconsistency label.</p>
            <div className="bg-white border border-black/30 p-2">
              <p className="text-[11px] font-semibold mb-1 text-muted-foreground">Date formats we understand:</p>
              <ul className="text-[11px] space-y-1 list-disc list-inside text-card-foreground">
                {DATE_PATTERNS.map(pattern => (
                  <li key={pattern}>{pattern}</li>
                ))}
              </ul>
            </div>
          </div>
        </AccordionCard>

        <AccordionCard title="TRAINING DATA">
          <div className="space-y-2 text-xs leading-relaxed text-muted-foreground">
            {DATA_FACTS.map(item => (
              <div key={item.label} className="bg-white border border-black/20 p-2 space-y-1">
                <p className="font-semibold text-sm text-black">{item.label}</p>
                <p>{item.value}</p>
              </div>
            ))}
            <div className="bg-muted border border-black p-2 space-y-1">
              <p className="text-[11px] font-semibold text-muted-foreground">Why twins?</p>
              <p>Every clean proposal sticks around in the batch. We only change one thing per twin, so the model learns that a single typo—not extra sentences—is all it takes to flip a label.</p>
            </div>
          </div>
        </AccordionCard>
      </div>
    </aside>
  )
}
