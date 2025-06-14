import { config } from 'dotenv';
config();

import '@/ai/flows/code-completion.ts';
import '@/ai/flows/code-summarization.ts';
import '@/ai/flows/code-assistant.ts';