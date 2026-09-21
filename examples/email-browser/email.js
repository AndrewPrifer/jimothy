// Keep the browser worker and the teacher request on the exact same input representation.
export function emailState(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      Object.keys(value).length !== 3 || ['from', 'subject', 'body'].some(key => typeof value[key] !== 'string')) {
    throw new Error('Provide from, subject, and body as text.');
  }
  if (!value.body.trim()) throw new Error('Enter an email to classify.');
  // Alphabetical keys match canonical-json-v1 in the training pipeline and Node SDK.
  const state = { body: value.body, from: value.from, subject: value.subject };
  if (JSON.stringify(state).length > 100_000) throw new Error('Use an email under 100,000 characters.');
  return state;
}

export const examples = [
  { name: 'Lunch', from: 'Maya <maya@example.com>', subject: 'Lunch on Friday?',
    body: 'Hi! Are you free for lunch on Friday? I can meet near your office around twelve. Let me know what works for you.\n\nMaya' },
  { name: 'Offer', from: 'Field Supply <offers@example.com>', subject: 'Weekend sale: 25% off',
    body: 'Our weekend sale is here. Save 25% on jackets, bags, and outdoor essentials through Sunday. Use code WEEKEND at checkout. Explore the collection and find your next favourite.' },
  { name: 'Security', from: 'Account Security <security@example.com>', subject: 'New sign-in to your account',
    body: 'We noticed a new sign-in to your account from a browser in Melbourne. If this was you, no action is needed. If you do not recognise this activity, review your recent sessions and change your password.' },
  { name: 'Mention', from: 'PhotoCircle <notifications@example.com>', subject: 'Alex tagged you in a photo',
    body: 'Alex Rivera tagged you in a photo on PhotoCircle. Sign in to see the photo, leave a comment, or manage the tag in your profile settings. You are receiving this notification because photo tags are enabled.' },
  { name: 'Discussion', from: 'Garden Forum <discussions@example.com>', subject: 'New replies: growing tomatoes in pots',
    body: 'There are three new replies to the community thread you follow, “Growing tomatoes in pots.” Sam recommends a deeper container, and Lee shared a watering schedule. Visit the discussion to reply or change your thread subscriptions.' },
  { name: 'Receipt', from: 'Paper Shop <orders@example.com>', subject: 'Receipt for order #1048',
    body: 'Thanks for your order. We received your payment of $24.00 for two notebooks. Your order number is 1048. We will send another email with tracking details when your package ships.' },
];
