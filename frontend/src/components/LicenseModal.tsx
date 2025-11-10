import { useState } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Copy, X } from 'lucide-react';

const APACHE_2_LICENSE = `Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Copyright 2025 Bala Subramanyam Duggirala, HDR Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.`;

interface LicenseModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function LicenseModal({ isOpen, onClose }: LicenseModalProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(APACHE_2_LICENSE);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <Card
        className="max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden border-2 border-black shadow-brutal"
        onClick={(e) => e.stopPropagation()}
      >
        <CardHeader className="flex flex-row items-center justify-between border-b-2 border-black">
          <CardTitle>Apache License 2.0</CardTitle>
          <button
            onClick={onClose}
            className="p-1 bg-destructive text-destructive-foreground hover:bg-red-600 border-2 border-black shadow-brutal rounded transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </CardHeader>
        <CardContent className="p-6 space-y-4">
          <div className="overflow-y-auto max-h-[60vh]">
            <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed">
              {APACHE_2_LICENSE}
            </pre>
          </div>
          <div className="flex justify-end border-t-2 border-black pt-4">
            <Button onClick={handleCopy} variant="secondary" size="sm">
              <Copy className="w-4 h-4" />
              {copied ? 'Copied!' : 'Copy License'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

