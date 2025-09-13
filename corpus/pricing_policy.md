# Vending Machine Pricing Policy

## Core Pricing Philosophy

This policy establishes the fundamental approach to pricing that maximizes revenue while maintaining customer satisfaction and competitive positioning. The vending machine bot should treat pricing as a dynamic tool that responds to market conditions, inventory levels, and customer behavior patterns.

## Base Pricing Structure

### Markup Categories
The bot should apply different markup percentages based on product categories, recognizing that different items have varying price elasticity and customer expectations:

**High-Demand Convenience Items** (sodas, energy drinks, popular snacks): Apply a 150-200% markup from wholesale cost. These items have low price sensitivity because customers value convenience highly.

**Specialty or Premium Items** (organic snacks, premium beverages, health foods): Apply a 100-150% markup. Customers purchasing these items are typically less price-sensitive and willing to pay for perceived quality.

**Bulk or Basic Items** (water, simple crackers, basic candy): Apply a 75-125% markup. These items compete more directly with retail stores, so pricing must remain reasonable.

### Dynamic Pricing Triggers

The bot should adjust prices based on specific conditions that indicate changing market dynamics:

**High Demand Periods**: Increase prices by 10-25% during peak hours (lunch time, after school, evening rush) when foot traffic and purchase urgency are highest. Monitor sales velocity to ensure price increases don't reduce total revenue.

**Low Inventory Situations**: When stock levels drop below 20% for popular items, implement scarcity pricing by increasing prices 15-30%. This both maximizes revenue from remaining units and naturally reduces demand to extend availability.

**Weather-Dependent Adjustments**: Increase beverage prices by 10-20% during hot weather (above 80°F) and decrease by 5-10% during cold weather (below 50°F). Apply reverse logic for hot items like coffee or soup.

**Competition Response**: Monitor nearby pricing when possible and maintain competitive positioning. Price 5-10% below direct competitors for identical items, or 10-15% above if location advantage exists (exclusive access, higher foot traffic).

## Psychological Pricing Strategies

### Price Point Psychology
Structure prices to leverage customer psychology and purchasing behavior:

Use charm pricing (ending in 9, 99, or 95) for items under $3.00 to create perception of value. For premium items above $3.00, use round numbers to convey quality and reduce perception of nickel-and-diming.

Implement price anchoring by positioning a few high-priced premium items prominently to make mid-tier items appear more reasonable by comparison.

### Bundle Pricing Opportunities
Create value perception through strategic bundling:

Offer combination deals (drink + snack) at a 10-15% discount from individual pricing to increase average transaction value. Ensure bundled items complement each other and have different peak demand times to optimize inventory turnover.

## Revenue Optimization Guidelines

### Margin Protection Rules
Never price below the following minimum margins except in specific clearance situations:

Maintain minimum 50% gross margin on all items to cover operational costs (electricity, maintenance, restocking labor, location fees). Calculate this margin from total cost including wholesale price, transaction fees, and allocated operational expenses.

For items approaching expiration, never discount below 25% gross margin unless the alternative is complete loss.

### Price Testing Framework
Implement systematic price optimization:

Test price changes on similar items or time periods to gather data on price elasticity. Increase prices gradually (5-10% increments) while monitoring sales volume to find optimal price points.

Track key metrics including total revenue per day, units sold per item, and profit margins. Adjust pricing when revenue trends show consistent improvement opportunities.

## Competitive Intelligence Integration

### Market Positioning
Maintain awareness of competitive landscape:

Position pricing to reflect location value proposition. Premium locations (airports, hospitals, office buildings) can sustain 20-40% higher prices than street-level machines due to convenience factor and limited alternatives.

Regular price surveys of nearby retail options help maintain realistic pricing that reflects genuine convenience premium rather than exploitation of captive customers.

### Customer Tolerance Monitoring
Track customer behavior to understand price sensitivity:

Monitor abandoned transactions (selections made but not purchased) as indicators of price resistance. High abandonment rates may indicate pricing above market tolerance.

Observe purchasing pattern changes following price adjustments to identify optimal pricing sweet spots for different customer segments and product categories.

## Implementation Notes for Bot Decision-Making

The vending machine bot should evaluate pricing decisions using this hierarchy: first ensure minimum margin requirements are met, then apply category-appropriate markup, then consider dynamic adjustments based on current conditions, and finally optimize for psychological impact and competitive positioning.

All pricing decisions should be logged with reasoning and results tracked to enable continuous improvement of the pricing algorithm. This data-driven approach ensures the bot learns from market feedback and becomes more profitable over time.