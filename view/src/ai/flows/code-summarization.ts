'use server';

/**
 * @fileOverview This file defines a Genkit flow for summarizing code.
 *
 * - summarizeCode - A function that takes code as input and returns a summary.
 * - SummarizeCodeInput - The input type for the summarizeCode function.
 * - SummarizeCodeOutput - The return type for the summarizeCode function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SummarizeCodeInputSchema = z.object({
  code: z.string().describe('The code to be summarized.'),
});
export type SummarizeCodeInput = z.infer<typeof SummarizeCodeInputSchema>;

const SummarizeCodeOutputSchema = z.object({
  summary: z.string().describe('A brief summary of the code.'),
});
export type SummarizeCodeOutput = z.infer<typeof SummarizeCodeOutputSchema>;

export async function summarizeCode(input: SummarizeCodeInput): Promise<SummarizeCodeOutput> {
  return summarizeCodeFlow(input);
}

const summarizeCodePrompt = ai.definePrompt({
  name: 'summarizeCodePrompt',
  input: {schema: SummarizeCodeInputSchema},
  output: {schema: SummarizeCodeOutputSchema},
  prompt: `You are an expert software developer. Please summarize the following code. Be concise and focus on the code's main functionality.\n\nCode:\n{{{code}}}`,
});

const summarizeCodeFlow = ai.defineFlow(
  {
    name: 'summarizeCodeFlow',
    inputSchema: SummarizeCodeInputSchema,
    outputSchema: SummarizeCodeOutputSchema,
  },
  async input => {
    const {output} = await summarizeCodePrompt(input);
    return output!;
  }
);
