from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.text import slugify

from core.models import ResearchSection, ResearchItem


RESEARCH_ITEMS = [
    {
        "title": "A Literature Review of the Seriousness of Flooding-based DoS Attack",
        "slug": "a-literature-review-of-the-seriousness-of-flooding-based-dos-attack",
        "research_type": "Conference Paper",
        "publish_date": "November 2022",
        "reads": "404",
        "citations": "2",
        "authors": [
            "Mohamed Albadi",
            "Moaiad Ahmad Khder",
            "Samah Fujo",
            "Tasneem Yousif",
        ],
        "primary_action": "Read More",
        "primary_action_href": "https://www.researchgate.net/publication/366717136_A_Literature_Review_of_the_Seriousness_of_Flooding-based_DoS_Attack",
        "share_href": "https://www.researchgate.net/publication/366717136_A_Literature_Review_of_the_Seriousness_of_Flooding-based_DoS_Attack",
        "external_image_url": "https://images.unsplash.com/photo-1510511459019-5dda7724fd87?auto=format&fit=crop&w=1200&q=80",
        "alt_text": "Cybersecurity network attack concept with dark digital interface",
        "sort_order": 1,
    },
    {
        "title": "Applying Machine Learning- Supervised Learning Techniques for Tennis Players Dataset Analysis",
        "slug": "applying-machine-learning-supervised-learning-techniques-for-tennis-players-dataset-analysis",
        "research_type": "Article",
        "publish_date": "November 2022",
        "reads": "933",
        "citations": "7",
        "authors": [
            "Moaiad Khder",
            "Samah Fujo",
        ],
        "primary_action": "Read More",
        "primary_action_href": "https://www.researchgate.net/publication/365867086_Applying_Machine_Learning-_Supervised_Learning_Techniques_for_Tennis_Players_Dataset_Analysis",
        "share_href": "https://www.researchgate.net/publication/365867086_Applying_Machine_Learning-_Supervised_Learning_Techniques_for_Tennis_Players_Dataset_Analysis",
        "external_image_url": "https://images.unsplash.com/photo-1542144582-1ba00456b5e3?auto=format&fit=crop&w=1200&q=80",
        "alt_text": "Tennis court and racket representing sports analytics research",
        "sort_order": 2,
    },
    {
        "title": "Measuring Scientific Research Trends Using an Online Scientific Research Management System (OSRMS) قياس اتجاهات البحث العلمي باستخدام نظام إلكتروني لإدارة البحوث العلمية",
        "slug": "measuring-scientific-research-trends-using-online-scientific-research-management-system-osrms",
        "research_type": "Article",
        "publish_date": "October 2022",
        "reads": "139",
        "citations": "0",
        "authors": [
            "Samer Shorman",
            "Samah Fujo",
            "Moaiad Khder",
        ],
        "primary_action": "Read More",
        "primary_action_href": "https://www.researchgate.net/publication/364302693_Measuring_Scientific_Research_Trends_Using_an_Online_Scientific_Research_Management_System_OSRMS_qyas_atjahat_albhth_allmy_bastkhdam_nzam_adart_albhth_allmy_alalktrwny",
        "share_href": "https://www.researchgate.net/publication/364302693_Measuring_Scientific_Research_Trends_Using_an_Online_Scientific_Research_Management_System_OSRMS_qyas_atjahat_albhth_allmy_bastkhdam_nzam_adart_albhth_allmy_alalktrwny",
        "external_image_url": "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=1200&q=80",
        "alt_text": "Research dashboard and analytics workspace",
        "sort_order": 3,
    },
    {
        "title": "Effect of Artificial intelligence in the field of games on humanity",
        "slug": "effect-of-artificial-intelligence-in-the-field-of-games-on-humanity",
        "research_type": "Conference Paper",
        "publish_date": "June 2022",
        "reads": "31",
        "citations": "4",
        "authors": [
            "Moaiad Ahmad Khder",
            "Abdulrahman Yusuf Bahar",
            "Samah Fujo",
        ],
        "primary_action": "Read More",
        "primary_action_href": "https://www.researchgate.net/publication/363875504_Effect_of_Artificial_intelligence_in_the_field_of_games_on_humanity",
        "share_href": "https://www.researchgate.net/publication/363875504_Effect_of_Artificial_intelligence_in_the_field_of_games_on_humanity",
        "external_image_url": "https://images.unsplash.com/photo-1511512578047-dfb367046420?auto=format&fit=crop&w=1200&q=80",
        "alt_text": "Gaming setup representing artificial intelligence in games",
        "sort_order": 4,
    },
    {
        "title": "How can blockchain revolutionize the health sector during Health Pandemics (Covid-19) in Kingdom of Bahrain",
        "slug": "how-can-blockchain-revolutionize-the-health-sector-during-health-pandemics-covid-19-in-kingdom-of-bahrain",
        "research_type": "Conference Paper",
        "publish_date": "June 2022",
        "reads": "16",
        "citations": "6",
        "authors": [
            "Maryam Salman AlSayed Abdulrahman",
            "Moaiad Ahmad Khder",
            "Basel J. Al Ali",
            "Samah Fujo",
        ],
        "primary_action": "Read More",
        "primary_action_href": "https://www.researchgate.net/publication/363874903_How_can_blockchain_revolutionize_the_health_sector_during_Health_Pandemics_Covid-19_in_Kingdom_of_Bahrain",
        "share_href": "https://www.researchgate.net/publication/363874903_How_can_blockchain_revolutionize_the_health_sector_during_Health_Pandemics_Covid-19_in_Kingdom_of_Bahrain",
        "external_image_url": "https://images.unsplash.com/photo-1584515933487-779824d29309?auto=format&fit=crop&w=1200&q=80",
        "alt_text": "Healthcare technology concept representing blockchain in health",
        "sort_order": 5,
    },
]


class Command(BaseCommand):
    help = "Seed Research section and existing research items from static website data."

    @transaction.atomic
    def handle(self, *args, **options):
        # Keep only one active Research section.
        ResearchSection.objects.filter(is_active=True).update(is_active=False)

        section = ResearchSection.objects.create(
            title="Research",
            description="Published papers, articles, and academic contributions",
            is_active=True,
        )

        created_count = 0
        updated_count = 0

        for item_data in RESEARCH_ITEMS:
            slug = item_data.get("slug") or slugify(item_data["title"])[:480]

            item, created = ResearchItem.objects.update_or_create(
                slug=slug,
                defaults={
                    "section": section,
                    "title": item_data["title"],
                    "research_type": item_data["research_type"],
                    "publish_date": item_data["publish_date"],
                    "reads": item_data["reads"],
                    "citations": item_data["citations"],
                    "authors": item_data["authors"],
                    "primary_action": item_data["primary_action"],
                    "primary_action_href": item_data["primary_action_href"],
                    "share_href": item_data["share_href"],
                    "external_image_url": item_data["external_image_url"],
                    "alt_text": item_data["alt_text"],
                    "sort_order": item_data["sort_order"],
                    "is_active": True,
                },
            )

            if created:
                created_count += 1
            else:
                updated_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Research seeded successfully. Created {created_count}, updated {updated_count}."
            )
        )
